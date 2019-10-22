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
    def __init__(self, f0, f1, v, t, k, num_graph=3, alpha=0.2, atrous_k=2):
        super(ClusterBlock, self).__init__()
        self.f0 = f0  # input dim
        self.f1 = f1  # output dim
        self.v = v  # number of nodes
        self.t = t  # length of input sequence
        self.k = k  # number of cluster
        self.num_graph = num_graph  # number of graphs
        self.atrous_k = atrous_k
        #  soft clustering
        self.cluster_dense = nn.ModuleList([nn.Linear(t * f0, k, bias=False) for _ in range(num_graph)]) # for each graph
        for i in range(num_graph):
            nn.init.xavier_normal_(self.cluster_dense[i].weight, gain=1.414)
        self.softmax = nn.Softmax(dim=-1)

        #  graph attention for every cluster of the all graph
        # for each atrous_offset, for each graph, for each cluster
        self.GC_layers = nn.ModuleList([nn.ModuleList([nn.ModuleList(
            [CGAT(f0, f1, t, v, attn_type='atrous', atrous_k=atrous_k, atrous_offset=ak) for _ in range(k)]) for _ in
            range(num_graph)]) for ak in range(atrous_k)])
        
        self.vt = Variable(torch.FloatTensor(range(self.v)), requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.layer_norm = nn.LayerNorm(f1)
        
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
        cluster = list(map(lambda cd: self.softmax(cd(xv)), self.cluster_dense)) #(b,v,k)
        gc_out = []
        for i in range(self.num_graph):
            cluster_outputs = []
            for j in range(self.k):
                gc = sum(self.GC_layers[ak][i][j](x,graphs[i]) for ak in range(self.atrous_k))
                #  (b*v, t*f1) * (b*v, 1) -> (b*v, t*f1) -> (b, v, t, f1)
                wgc = (gc.view(b * self.v, self.t * self.f1) * cluster[i][:, :, j].view(b * self.v, 1)).\
                    view(b, self.v, self.t, self.f1)
                cluster_outputs.append(wgc)
            #  (b, v, t, f1)
            all_gc_out = torch.sum(torch.stack(cluster_outputs, dim=-1), dim=-1)
            gc_out.append(all_gc_out)
        #  the mean of three graphs, (b, v, t, f1)
        gc_act = torch.mean(torch.stack(gc_out, dim=-1), dim=-1)
        return self.layer_norm(gc_act), cluster


#  multi-head attention
class MultiHeadAttention(nn.Module):
    """
    A multi-head attention module.
    f0 = (q_0,k_0,v_0)
    f_hid = (q_hid,k_hid,v_hid)
    f1 = f_v_out
    """
    def __init__(self, f0, f_hid, f1, alpha=0.2):
        super().__init__()
        self.f0 = f0
        self.f1 = f1
        self.f_hid = f_hid
        
        self.q_0 = f0[0]
        self.k_0 = f0[1]
        self.v_0 = f0[2]
        
        self.q_h = f_hid[0]
        self.k_h = f_hid[1]
        self.v_h = f_hid[2]
        assert self.q_h == self.k_h
        
        self.alpha = alpha
        self.leakyrelu = nn.LeakyReLU(alpha)

        self.w_qs = nn.Linear(self.q_0, self.q_h, bias=False)
        self.w_ks = nn.Linear(self.k_0, self.k_h, bias=False)
        self.w_vs = nn.Linear(self.v_0, self.v_h, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (self.q_0+self.q_h)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (self.k_0+self.k_h)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (self.v_0+self.v_h)))

        self.attention = ScaledDotProductAttention(temperature=np.power(self.q_h, 0.5))
        self.fc = nn.Linear(self.v_h, self.f1, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(f1)
        
    def forward(self, q, k, v, mask=None):
        """
        MultiHeadAttention operation
        :param q: (b, t, d_q)
        :param k: (b, t, d_k)
        :param v: (b, v, t, d_v)
        :param mask:
        :return: (b, v, t, f1)
        """
        b_q, t_q, f_q = q.size()
        b_k, t_k, f_k = k.size()
        b_v, v_v, t_v, f_v = v.size()
        assert b_q == b_k == b_v
        assert f_q == self.q_0
        assert f_k == self.k_0
        assert f_v == self.v_0
        output, attn = self.attention(self.w_qs(q), self.w_ks(k), self.w_vs(v), mask=mask)
        output = output.view(b_v,v_v,t_q,self.v_h).contiguous()
        output = self.leakyrelu(self.fc(output))
        return self.layer_norm(output), attn


class PositionWiseFeedForward(nn.Module):
    """
    A position wise feed forward module.
    Project the dim of input to d_hid, and then to d_in back.
    """
    def __init__(self, d_in, d_hid,d_out, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.w_1 = nn.Linear(d_in, d_hid, bias=False)
        self.w_2 = nn.Linear(d_hid, d_out, bias=False)

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
    def __init__(self, f0, f_hid, f1, dropout=0.5):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(f0=(f0,f0,f0), f_hid=(f_hid,f_hid,f_hid), f1 = f1)
        self.pos_ffn = PositionWiseFeedForward(d_in = f1,d_hid = f1*2,d_out = f1)

    def forward(self, enc_input_qk, enc_input_v, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input_qk, enc_input_qk, enc_input_v, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """
    Decoder layer.
    Compose with three layers: multi-head attention, enc-dec attention and ffc.
    """
    def __init__(self, f0, enc_f0, f_hid, f1,dropout=0.5):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(f0=(f0,f0,f0),f_hid = (f_hid,f_hid,f_hid),f1=f_hid)
        self.enc_attn = MultiHeadAttention(f0=(f_hid,enc_f0,enc_f0),f_hid = (f_hid,f_hid,f_hid),f1=f1)
        self.pos_ffn = PositionWiseFeedForward(d_in = f1,d_hid = f1*2,d_out = f1)

    def forward(self, dec_input_qk, dec_input_v, enc_output_qk, enc_output_v, slf_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input_qk, dec_input_qk, dec_input_v, mask=slf_attn_mask)
        #  Flat the dim v
        dec_output_flat = torch.mean(dec_output, 1)
        dec_output, dec_enc_attn = self.enc_attn(dec_output_flat, enc_output_qk, enc_output_v, mask=None)
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
    io_list = [[input,gc_out/attn_in,attn_hid,attn_out],...,[]]
    io_list=[[5,8,16,16],[16,32,32,16],[16,16,16,16]]
    toy example: io_list=[[5,8,8,8],[8,8,8,8]]
    """
    def __init__(self, v, t, k_cluster, n_layers, io_list, pe_dim=5, alpha=0.2):
        super().__init__()
        assert n_layers == len(io_list)
        assert io_list[0][0] == pe_dim
        
        #  generate input-output dim list of multi layers.
        self.pe_dim = pe_dim
        self.io_list = io_list
        self.n_layers = n_layers
        #  cluster graph attention module stack.
        self.cluster_block = nn.ModuleList(
            list(map(lambda x: ClusterBlock(x[0], x[1], v, t, k_cluster), self.io_list))
        )
        #  encoder layer module stack.
        self.layer_stack = nn.ModuleList(
            list(map(lambda x: EncoderLayer(x[1],x[2],x[3]), self.io_list))
        )
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x_d, x_pe, graph, return_attns=False):
        """
        :param x_d: input signal, shape: (b, v, t, f)
        :param x_pe: position encoding, shape: (b, t, 5)
        :param graph: list of adjacent matrix
        :param return_attns: boolean
        :return: (b, v, t, f)
        """
        b, v, t, f = x_d.size()
        enc_slf_attn_list = []
        cluster_mat = []

        #  get position encoding, pe: (B,T) f 0-4 for periodicity (the sin/cos functions), 5 for closeness (the exp function)

        inputs = x_d.repeat(1, 1, 1, self.pe_dim)  # (b, v, t, 5)
        inputs = (inputs.permute(1, 0, 2, 3).contiguous() + x_pe).permute(1, 0, 2, 3).contiguous()  # (b, v, t, f)

        enc_output = inputs
        for i in range(0, self.n_layers):
            ccb_layer = self.cluster_block[i]
            enc_layer = self.layer_stack[i]
            enc_output, cluster = ccb_layer(enc_output, graph)
            cluster_mat.append(cluster)
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
    io_list=[[5,8,16,16],[16,8,8,8],[8,8,8,8]]
    toy example: io_list=[[5,8,8,8],[8,8,8,8]]
    """

    def __init__(self, v, t, k_cluster, n_layers, io_list, enc_f, pe_dim=5, alpha=0.2):
        super().__init__()
        assert n_layers == len(io_list)

        #  generate input-output dim list of multi layers.
        self.io_list = io_list
        self.pe_dim = pe_dim
        self.n_layers = n_layers
        self.enc_f = enc_f
        #  cluster graph attention module stack.
        self.cluster_block = nn.ModuleList(
            list(map(lambda x: ClusterBlock(x[0], x[1], v, t, k_cluster), self.io_list))
        )
        #  decoder layer module stack.
        self.layer_stack = nn.ModuleList(
            list(map(lambda x: DecoderLayer(x[1],self.enc_f,x[2],x[3]),self.io_list))
        )
        self.leakyrelu = nn.LeakyReLU(alpha)

        
    def forward(self, x_d, x_pe, enc_output, graph, return_attns=False):
        """
        :param x_d: target signal, shape: (b, v, t, f)
        :param x_pe: position encoding, shape: (b, t, 5)
        :param enc_output: the output of encoder, shape: (b, v, t, f)
        :param graph: list of adjacent matrix
        :param return_attns: boolean
        :return: (b, v, t, f)
        """
        b, v, t, f = x_d.size()
        cluster_mat = []
        dec_slf_attn_list = []
        dec_enc_attn_list = []

        #  get position encoding, pe: (B,T) f 0-4 for periodicity (the sin/cos functions), 5 for closeness (the exp function)
        inputs = x_d.repeat(1, 1, 1, self.pe_dim)  # (b, v, t, 5)
        inputs = (inputs.permute(1, 0, 2, 3).contiguous() + x_pe).permute(1, 0, 2, 3).contiguous()  # (b, v, t, f)

        enc_output_qk = torch.mean(enc_output, dim=1)
        enc_output_v = enc_output
        slf_attn_mask_subseq = get_subsequent_mask(x_d)  # b,t,t

        dec_output = inputs
        for i in range(0, self.n_layers):
            ccb_layer = self.cluster_block[i]
            dec_layer = self.layer_stack[i]
            
            dec_output, cluster = ccb_layer(dec_output, graph)
            cluster_mat.append(cluster)
            dec_output = dec_output.view(b, v, t, -1)
            dec_output_flat = torch.mean(dec_output, dim=1)
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output_flat, dec_output,
                                                               enc_output_qk, enc_output_v, slf_attn_mask_subseq)
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
    def __init__(self, v, x_t, y_t, k_cluster, n_layers, output_dim=1, pe_dim=5, alpha=0.2):
        super(CGT, self).__init__()
        self.v = v
        
        self.enc_io_list = [[5,8,16,16],[16,32,32,16],[16,16,16,16]]
        self.dec_io_list = [[5,8,16,16],[16,16,16,16],[16,8,8,8]]
        
        self.encoder = Encoder(v=v, t=x_t, k_cluster=k_cluster, io_list = self.enc_io_list, n_layers=n_layers, pe_dim=pe_dim, alpha=alpha)

        self.decoder = Decoder(v=v, t=y_t, k_cluster=k_cluster, io_list = self.dec_io_list, enc_f = self.enc_io_list[-1][-1], n_layers=n_layers, pe_dim=pe_dim, alpha=alpha)

        self.hid_dim = self.decoder.io_list[-1][-1]
        self.aggregate = nn.Linear(self.hid_dim, output_dim,bias=True)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x_d, x_pe, y_d, y_pe, graph):
        """
        :param x_d: (b, v, x_t, f_in)
        :param x_pe: (b, x_t, 5)
        :param y_d: (b, v, y_t, f_out)
        :param y_pe: (b, y_t, 5)
        :param graph: [num_graph, v, v]
        :return: (b, v, y_t, f_out)
        """
        enc_output, enc_cluster, *_ = self.encoder(x_d, x_pe, graph)
        dec_output, dec_cluster, *_ = self.decoder(y_d, y_pe, enc_output, graph)
        dec_output = self.leakyrelu(self.aggregate(dec_output))
        return dec_output, enc_cluster + dec_cluster
