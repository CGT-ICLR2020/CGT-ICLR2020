import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


#  clustered graph attention
class CGAT(nn.Module):
    def __init__(self, f0, f1, t, alpha=0.2, attn_type='atrous', atrous_k=2, atrous_offset=0):
        super(CGAT, self).__init__()

        self.f0 = f0
        self.f1 = f1
        self.alpha = alpha
        self.attn_type = attn_type
        self.atrous_k = atrous_k
        self.atrous_offset = atrous_offset

        assert attn_type in ['dense', 'atrous']

        self.W = nn.Linear(f0, f1, bias=True)
        nn.init.xavier_normal_(self.W.weight, gain=1.414)

        # squeeze b*t
        self.Wt = nn.Linear(t, 1, bias=False)
        nn.init.xavier_normal_(self.Wt.weight, gain=1.414)

        self.a = nn.Linear(2 * f1, 1, bias=False)
        nn.init.xavier_normal_(self.a.weight, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        """
        graph attention
        :param input: the input signal, shape: (b, v, t, f0)
        :param adj: the adjacency matrix of graph, shape: (v, v)
        :return: shape: (b, v, t, f1)
        """
        b, v, t, f0 = input.size()[0], input.size()[1], input.size()[2], input.size()[3]
        assert f0 == self.f0

        #  (b, v, t, f0) * (f0, f1) -> (b, v, t, f1)
        h = self.leakyrelu(self.W(input))

        #  (b, v, t, f1) -> (v, f1, b, t) -> (v, f1, b, 1) -> (v, f1, b) -> (v, f1)
        ht = torch.mean(self.Wt(h.permute(1, 3, 0, 2).contiguous()).squeeze(-1), dim=-1, keepdim=False)

        if self.attn_type == 'dense':
            #  (v, f1) -> (1, v, f1) -> (v, v, f1)
            ht1 = ht.unsqueeze(0).repeat(v, 1, 1)
            #  (v, f1) -> (v, 1, f1) -> (v, v, f1)
            ht2 = ht.unsqueeze(1).repeat(1, v, 1)
            a_input = torch.cat([ht1, ht2], dim=-1)  # (v, v, 2*f1)

            #  (v, v, 2*f1) * (2*f1, 1 ) -> (v, v)
            e = self.leakyrelu(self.a(a_input).squeeze(-1))
            zero_vec = -9e15 * torch.ones_like(adj).to('cuda')  # (v, v)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=-1)  # (v, v)

            #  (b, t, f1, v) * (v, v)  ->  (b, t, f1, v) -> (b, v, t, f)
            h_prime = torch.matmul(h.permute(0, 2, 3, 1), attention.permute(1, 0)).permute(0, 3, 1, 2).contiguous()

        if self.attn_type == 'atrous':
            va = int((v - self.atrous_offset - 1) / self.atrous_k) + 1
            #  (b, va, t, f1)
            ha = h[:, self.atrous_offset::self.atrous_k]
            #  (va, f1) -> (1, va, f1) -> (v,  va, f1)
            ht1 = ht[self.atrous_offset::self.atrous_k].unsqueeze(0).repeat(v, 1, 1)
            #  (v, 1, f1) -> (v, va, f1)
            ht2 = ht.unsqueeze(1).repeat(1, va, 1)
            a_input = torch.cat([ht1, ht2], dim=-1)  # (v, va, 2*f1)

            #  (v, va, 2*f1) * (2*f1, 1) -> (v, va, 1) -> (v, va)
            e = self.leakyrelu(self.a(a_input).squeeze(-1))
            adj_c = adj[:, self.atrous_offset::self.atrous_k]
            zero_vec = -9e15 * torch.ones_like(adj_c).to('cuda')
            attention = torch.where(adj_c > 0, e, zero_vec)  # (v, va)
            attention = F.softmax(attention, dim=-1)

            #  (b, t, f1, va) * (va, v) -> (b, t, f1, v) -> (b, v, t, f1)
            h_prime = torch.matmul(ha.permute(0, 2, 3, 1), attention.permute(1, 0)).permute(0, 3, 1, 2).contiguous()

        return self.leakyrelu(h_prime)


class ClusterBlock(nn.Module):
    def __init__(self, f0, f1, v, t, k, num_graph=3, alpha=0.2):
        super(ClusterBlock, self).__init__()
        self.f0 = f0  # input dim
        self.f1 = f1  # output dim
        self.v = v  # number of nodes
        self.t = t  # length of input sequence
        self.k = k  # number of cluster
        self.num_graph = num_graph  # number of graphs

        #  soft clustering
        self.cluster_dense = nn.ModuleList([nn.Linear(t * f0, k, bias=False) for _ in range(num_graph)])
        for i in range(num_graph):
            nn.init.xavier_normal_(self.cluster_dense[i].weight, gain=1.414)
        self.softmax = nn.Softmax(dim=-1)

        #  graph attention for every cluster of the all graph
        self.GC_layers1 = nn.ModuleList([nn.ModuleList(
            [CGAT(f0, f1, t, v, attn_type='atrous', atrous_k=2, atrous_offset=0) for _ in range(k)]) for _ in
            range(num_graph)])
        self.GC_layers2 = nn.ModuleList([nn.ModuleList(
            [CGAT(f0, f1, t, v, attn_type='atrous', atrous_k=2, atrous_offset=1) for _ in range(k)]) for _ in
            range(num_graph)])

        self.vt = Variable(torch.FloatTensor(range(self.v)), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, graphs):
        """

        :param x: (b, v, t, f0)
        :param graphs: List[(v,v)]
        :return: gc_act: (b, v, t, f1)
                 cluster: [num_graphs, (b, v, k)]
        """
        b = x.size()[0]

        # soft clustering
        xv = x.view(b, self.v, self.t * self.f0)
        cluster = list(map(lambda cd: self.softmax(cd(xv)), self.cluster_dense))

        assert len(self.GC_layers1) == len(graphs)
        assert len(self.GC_layers2) == len(graphs)
        gc_out = []
        for i in range(self.num_graph):
            cluster_outputs = []
            for j in range(self.k):
                gc1 = self.GC_layers1[i][j](x, graphs[i])
                gc2 = self.GC_layers2[i][j](x, graphs[i])
                gc = gc1 + gc2
                #  (b*v, t*f1) * (b*v, 1) -> (b*v, t*f1) -> (b, v, t, f1)
                wgc = (gc.view(b * self.v, self.t * self.f1) * cluster[i][:, :, j].view(b * self.v, 1)).\
                    view(b, self.v, self.t, self.f1)
                cluster_outputs.append(wgc.unsqueeze(-1))
            #  (b, v, t, f1)
            all_gc_out = torch.sum(torch.stack(cluster_outputs, dim=-1), dim=-1).squeeze(-1)
            gc_out.append(all_gc_out)
        #  the mean of three graphs, (b, v, t, f1)
        gc_act = torch.mean(torch.stack(gc_out, dim=-1), dim=-1)

        return gc_act, cluster


#  multi-head attention
class MultiHeadAttention(nn.Module):
    """
    A multi-head attention module.
    """
    def __init__(self, n_head, d_model_in, d_model_out, d_k, d_v, alpha=0.2):
        super().__init__()
        assert d_model_out == n_head * d_k
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(alpha)

        self.w_qs = nn.Linear(d_model_out, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model_in, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model_in, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model_in + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model_in + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model_in + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.fc = nn.Linear(n_head * d_v, d_model_out, bias=False)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, mask=None):
        """
        MultiHeadAttention operation
        :param q: (b, t, d_q)
        :param k: (b, t, d_k)
        :param v: (b, v, t, d_v)
        :param mask:
        :return: (b, v, t, d_model_out)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b_q, len_q, d0_q = q.size()
        sz_b_k, len_k, d0_k = k.size()
        sz_b_v, n_v, len_v, d0_v = v.size()

        #  multi-head attention
        q = self.w_qs(q).view(sz_b_q, len_q, -1).contiguous().view(sz_b_q, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b_k, len_k, -1).contiguous().view(sz_b_k, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b_v, n_v, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (b * nhead) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (b * nhead) x lk x dk
        v = v.permute(3, 0, 1, 2, 4).contiguous().view(-1, n_v, len_v, d_v)  # (n_head*b) x n_v x lv x dv
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b_v, n_v, len_q, d_v)
        output = output.permute(1, 2, 3, 4, 0).contiguous().view(sz_b_v, n_v, len_q, n_head * d_v)

        output = self.leakyrelu(self.fc(output))

        return output, attn


class PositionWiseFeedForward(nn.Module):
    """
    A position wise feed forward module.
    Project the dim of input to d_hid, and then to d_in back.
    """
    def __init__(self, d_in, d_hid, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.w_1 = nn.Linear(d_in, d_hid, bias=False)
        self.w_2 = nn.Linear(d_hid, d_in, bias=False)

        nn.init.xavier_normal_(self.w_1.weight, gain=1.414)
        nn.init.xavier_normal_(self.w_2.weight, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x):
        output = self.leakyrelu(self.w_2(self.leakyrelu(self.w_1(x))))
        return output


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        q: (b, t, f)
        k: (b, t, f)
        v: (b, t, v, f)
        :return: (b, n, t, f)
        """
        b, n, t, f = v.size()

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)  # fill the place with -inf where mask tensor is 1.

        attn = self.softmax(attn)
        _, d, t = attn.size()

        vr = v.permute(0, 2, 3, 1).contiguous().view(b, t, n*f)
        output = torch.bmm(attn, vr)
        output = output.contiguous().view(b, d, f, n).permute(0, 3, 1, 2)

        return output, attn


# transformer
class EncoderLayer(nn.Module):
    """
    Encoder layer.
    Compose with two layers: multi-head attention and ffc.
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.5):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_model, d_k, d_v)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner)

    def forward(self, enc_input_qk, enc_input_v, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input_qk, enc_input_qk, enc_input_v, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """
    Decoder layer.
    Compose with three layers: multi-head attention, enc-dec attention and ffc.
    """

    def __init__(self, enc_f, y_f, d_inner, n_head, d_k, d_v, dropout=0.5):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, y_f, y_f, d_k, d_v)
        self.enc_attn = MultiHeadAttention(n_head, enc_f, y_f, d_k, d_v)
        self.pos_ffn = PositionWiseFeedForward(y_f, d_inner)

    def forward(self, dec_input_qk, dec_input_v, enc_output, slf_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input_qk, dec_input_qk, dec_input_v, mask=slf_attn_mask)
        #  Flat the dim v
        dec_output_flat = torch.mean(dec_output, 1)
        enc_output_flat = torch.mean(enc_output, 1)

        dec_output, dec_enc_attn = self.enc_attn(dec_output_flat, enc_output_flat, enc_output, mask=None)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


def get_subsequent_mask(seq):
    """
    A subsequent mask for masking out the subsequent info.
    seq: (b, v, t, f)
    """
    sz_b, _, len_s = seq.size()[:-1]
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # (b, t, t)

    return subsequent_mask


class Encoder(nn.Module):
    """
    The encoder with self attention mechanism of transformer.
    """
    def __init__(self, v, t, k_cluster, n_layers, n_head, d_inner, out_dim_list, pe_dim=5, alpha=0.2):
        super().__init__()
        assert n_layers == len(out_dim_list)

        #  generate input-output dim list of multi layers.
        self.io_list = [[pe_dim - 1, pe_dim - 1]]
        self.tf_io_list = [[pe_dim - 1, out_dim_list[0]]]
        for i in range(1, n_layers):
            self.io_list.append([out_dim_list[i-1], out_dim_list[i]])
            self.tf_io_list.append([out_dim_list[i - 1], out_dim_list[i]])

        self.n_layers = n_layers
        #  cluster graph attention module stack.
        self.cluster_block = nn.ModuleList(
            list(map(lambda x: ClusterBlock(x[0], x[1], v, t, k_cluster), self.io_list))
        )
        self.pe_dim = pe_dim
        self.pos_enc_qk = nn.Linear(pe_dim, self.io_list[1][0])
        self.pos_enc_v = nn.Linear(pe_dim, self.io_list[1][0])
        self.leakyrelu = nn.LeakyReLU(alpha)
        #  encoder layer module stack.
        self.layer_stack = nn.ModuleList(
            list(map(lambda x: EncoderLayer(x[1], d_inner, n_head, x[1] // n_head, x[1] // n_head),
                     self.tf_io_list))
        )

    def forward(self, x_d, x_pe, graph, return_attns=False):
        """
        :param x_d: input signal, shape: (b, v, t, f)
        :param x_pe: position encoding, shape: (b, t, 6)
        :param graph: list of adjacent matrix
        :param return_attns: boolean
        :return: (b, v, t, f)
        """
        sz_b, v, t, f = x_d.size()
        enc_slf_attn_list = []
        cluster_mat = []

        #  get position encoding, pe: (B,T) f 0-3 periodicty, 4 trend qk, 5 trend v
        qk_ = x_pe[:, :, -2].unsqueeze(-1)
        v_ = x_pe[:, :, -1].unsqueeze(-1)
        x_qk = torch.mean(x_d, dim=1) + qk_
        x_v = (x_d.permute(1, 0, 2, 3).contiguous() + v_).permute(1, 0, 2, 3).contiguous()

        inputs = x_d.repeat(1, 1, 1, self.pe_dim - 1)  # (b, v, t, 4)
        periodicity = x_pe[:, :, :4]
        inputs = (inputs.permute(1, 0, 2, 3).contiguous() + periodicity).\
            permute(1, 0, 2, 3).contiguous()  # (b, v, t, f)

        #  cluster graph attention
        inputs, cluster = self.cluster_block[0](inputs, graph)  # (b,v,t,4), (b,v,k)
        cluster_mat.append(cluster)
        #  flat the dim v
        inputs_flat = torch.mean(inputs, dim=1)  # (b, t, f)
        #  encoder layer
        enc_input_qk = torch.cat([inputs_flat, x_qk], dim=-1)
        enc_input_v = torch.cat([inputs, x_v], dim=-1)
        enc_input_qk = self.leakyrelu(self.pos_enc_qk(enc_input_qk))
        enc_input_v = self.leakyrelu(self.pos_enc_v(enc_input_v))
        enc_output, enc_slf_attn = self.layer_stack[0](enc_input_qk, enc_input_v)  # (b, v, t, f2)
        if return_attns:
            enc_slf_attn_list += [enc_slf_attn]

        for i in range(1, self.n_layers):
            ccb_layer = self.cluster_block[i]
            enc_layer = self.layer_stack[i]
            enc_output, cluster = ccb_layer(enc_output, graph)
            cluster_mat.append(cluster)
            enc_output = enc_output.view(sz_b, v, t, -1)
            enc_output_flat = torch.mean(enc_output, dim=1)
            enc_output, enc_slf_attn = enc_layer(enc_output_flat, enc_output)

            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list, cluster_mat
        return enc_output, cluster_mat


class Decoder(nn.Module):
    """
    The decoder with attention mechanism of transformer.
    """

    def __init__(self, v, t, k_cluster, n_layers, n_head, d_inner, out_dim_list, enc_f=8, y_d_f=1, pe_dim=5, alpha=0.2):
        super().__init__()
        assert n_layers == len(out_dim_list)

        #  generate input-output dim list of multi layers.
        self.io_list = [[pe_dim - 1, pe_dim - 1]]
        self.tf_io_list = [[pe_dim - 1, out_dim_list[0]]]
        for i in range(1, n_layers):
            self.io_list.append([out_dim_list[i - 1], out_dim_list[i]])
            self.tf_io_list.append([out_dim_list[i - 1], out_dim_list[i]])

        self.n_layers = n_layers
        #  cluster graph attention module stack.
        self.cluster_block = nn.ModuleList(
            list(map(lambda x: ClusterBlock(x[0], x[1], v, t, k_cluster), self.io_list))
        )
        self.enc_f = enc_f
        self.y_d_f = y_d_f
        self.pe_dim = pe_dim
        self.pos_dec_qk = nn.Linear(pe_dim, self.io_list[1][0])
        self.pos_dec_v = nn.Linear(pe_dim, self.io_list[1][0])
        self.leakyrelu = nn.LeakyReLU(alpha)
        #  decoder layer module stack.
        self.layer_stack = nn.ModuleList(
            list(map(
                lambda x: DecoderLayer(enc_f, x[1], d_inner, n_head, x[1] // n_head, x[1] // n_head),
                self.tf_io_list))
        )

    def forward(self, x_d, x_pe, enc_output, graph, return_attns=False):
        """
        :param x_d: target signal, shape: (b, v, t, f)
        :param x_pe: position encoding, shape: (b, t, 6)
        :param enc_output: the output of encoder, shape: (b, v, t, f)
        :param graph: list of adjacent matrix
        :param return_attns: boolean
        :return: (b, v, t, f)
        """
        b, v, t, f = x_d.size()
        cluster_mat = []
        dec_slf_attn_list = []
        dec_enc_attn_list = []

        #  get position encoding, pe: (B,T) f 0-3 periodicty, 4 trend qk, 5 trend v
        qk_ = x_pe[:, :, -2].unsqueeze(-1)
        v_ = x_pe[:, :, -1].unsqueeze(-1)
        x_qk = torch.mean(x_d, dim=1) + qk_
        x_v = (x_d.permute(1, 0, 2, 3).contiguous() + v_).permute(1, 0, 2, 3).contiguous()

        inputs = x_d.repeat(1, 1, 1, self.pe_dim - 1)  # (b, v, t, 4)
        periodicity = x_pe[:, :, :4]
        inputs = (inputs.permute(1, 0, 2, 3).contiguous() + periodicity).\
            permute(1, 0, 2, 3).contiguous()  # (b, v, t, f)

        #  cluster graph attention
        inputs, cluster = self.cluster_block[0](inputs, graph)  # (b,v,t,4), (b,n,k)
        cluster_mat.append(cluster)
        #  flat the dim v
        inputs_flat = torch.mean(inputs, dim=1)  # (b, t, f)
        #  decoder layer
        dec_input_qk = torch.cat([inputs_flat, x_qk], dim=-1)
        dec_input_v = torch.cat([inputs, x_v], dim=-1)
        dec_input_qk = self.leakyrelu(self.pos_dec_qk(dec_input_qk))
        dec_input_v = self.leakyrelu(self.pos_dec_v(dec_input_v))

        slf_attn_mask_subseq = get_subsequent_mask(x_d)  # b,t,t
        dec_output, dec_slf_attn, dec_enc_attn = self.layer_stack[0](dec_input_qk, dec_input_v,
                                                                     enc_output, slf_attn_mask_subseq)
        if return_attns:
            dec_slf_attn_list += [dec_slf_attn]
            dec_enc_attn_list += [dec_enc_attn]

        for i in range(1, self.n_layers):
            ccb_layer = self.cluster_block[i]
            dec_layer = self.layer_stack[i]

            dec_output, cluster = ccb_layer(dec_output, graph)
            cluster_mat.append(cluster)
            dec_output = dec_output.view(b, v, t, -1)
            dec_output_flat = torch.mean(dec_output, dim=1)

            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output_flat, dec_output,
                                                               enc_output, slf_attn_mask_subseq)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output, cluster_mat


class CGT(nn.Module):
    """
    A clustered graph transformer model for spatio-temporal predicting.
    """
    def __init__(self, v, x_t, y_t, k_cluster, d_inner, n_layers,
                 n_head_enc, n_head_dec, out_dim_list, output_dim=1, pe_dim=5, alpha=0.2):
        super(CGT, self).__init__()
        self.v = v
        self.encoder = Encoder(v=v, t=x_t, k_cluster=k_cluster, n_layers=n_layers, n_head=n_head_enc, d_inner=d_inner,
                               out_dim_list=out_dim_list, pe_dim=pe_dim, alpha=alpha)

        self.decoder = Decoder(v=v, t=y_t, k_cluster=k_cluster, n_layers=n_layers, n_head=n_head_dec, d_inner=d_inner,
                               out_dim_list=out_dim_list, enc_f=out_dim_list[-1], y_d_f=output_dim, pe_dim=pe_dim, alpha=alpha)

        self.hid_dim = self.decoder.io_list[-1][-1]
        self.aggregate = nn.Linear(self.hid_dim, output_dim)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x_d, x_pe, y_d, y_pe, graph):
        """
        :param x_d: (b, v, x_t, f_in)
        :param x_pe: (b, x_t, 6)
        :param y_d: (b, v, y_t, f_out)
        :param y_pe: (b, y_t, 6)
        :param graph: [num_graph, v, v]
        :return: (b, v, y_t, f_out)
        """
        enc_output, enc_cluster, *_ = self.encoder(x_d, x_pe, graph)
        dec_output, dec_cluster, *_ = self.decoder(y_d, y_pe, enc_output, graph)
        dec_output = self.leakyrelu(self.aggregate(dec_output))
        return dec_output, enc_cluster + dec_cluster
