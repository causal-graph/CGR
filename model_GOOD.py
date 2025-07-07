from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv
from gcn_conv import GCNConv
import random
import pdb


class Causal(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(
        self,
        num_features,
        num_classes,
        args,
        gfn=False,
        edge_norm=True,
    ):
        super(Causal, self).__init__()
        hidden = args.hidden
        num_conv_layers = args.layers
        self.args = args
        self.global_pool = global_add_pool
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        hidden_in = num_features
        self.num_classes = num_classes
        #  self.with_random = args.with_random
        hidden_out = num_classes
        # self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden,
                                 gfn=True)  # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(
                GINConv(
                    Sequential(Linear(hidden, hidden), BatchNorm1d(hidden),
                               ReLU(), Linear(hidden, hidden), ReLU())))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno = BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, xo, xc):

        xc_logis = self.context_readout_layer(xc)
        xco_logis = self.random_readout_layer(xc, xo)
        xo_logis = self.objects_readout_layer(xo)
        return xc_logis, xo_logis, xco_logis

    def forward_xo(self, data):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        #  edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        #  node_weight_c = node_att[:, 0]
        node_weight_o = node_att[:, 1]

        #   xc = node_weight_c.view(-1, 1) * x
        xo = node_weight_o.view(-1, 1) * x
        # xc = F.relu(self.context_convs(self.bnc(xc), edge_index,
        #                                edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index,
                                       edge_weight_o))

        # xc = self.global_pool(xc, batch)  #short cut feature(batch_size*hidden)
        xo = self.global_pool(xo, batch)  #causal feature (batch_size*hidden)

        # xc_logis = self.context_readout_layer(xc)
        # xco_logis = self.random_readout_layer(xc, xo)
        # # return xc_logis, xo_logis, xco_logis
        # xo_logis = self.objects_readout_layer(xo, train_type)
        return xo

    def eval_forward(self, data):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        #  edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        #  node_weight_c = node_att[:, 0]
        node_weight_o = node_att[:, 1]

        #   xc = node_weight_c.view(-1, 1) * x
        xo = node_weight_o.view(-1, 1) * x
        # xc = F.relu(self.context_convs(self.bnc(xc), edge_index,
        #                                edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index,
                                       edge_weight_o))

        #    xc = self.global_pool(xc, batch)#short cut feature(batch_size*hidden)
        xo = self.global_pool(xo, batch)  #causal feature (batch_size*hidden)

        xo_logis = self.objects_readout_layer(xo)
        return xo_logis

    def forward_xc(self, data):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        # edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight_c = node_att[:, 0]
        # node_weight_o = node_att[:, 1]

        xc = node_weight_c.view(-1, 1) * x
        #xo = node_weight_o.view(-1, 1) * x
        xc = F.relu(self.context_convs(self.bnc(xc), edge_index,
                                       edge_weight_c))
        # xo = F.relu(self.objects_convs(self.bno(xo), edge_index,
        #edge_weight_o))

        xc = self.global_pool(xc, batch)  #short cut feature(batch_size*hidden)

        return xc

    # def _dequeue_and_enqueue(self, keys):
    #     # gather keys before updating queue
    #     batch_size = keys.shape[0]

    #     ptr = int(self.queue_ptr)
    #     # assert self.K % batch_size == 0  # for simplicity

    #     # replace the keys at ptr (dequeue and enqueue)
    #     self.queue[:, ptr:ptr + batch_size] = keys.T
    #     ptr = (ptr + batch_size) % self.K  # move pointer

    #     self.queue_ptr[0] = ptr

    def context_readout_layer(self, x):

        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis

    def objects_readout_layer(self, x):

        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        x_logis =  F.log_softmax(x, dim=-1)

        return x_logis

    def random_readout_layer(self, xc, xo):
        '''
        num = xc.shape[0]
        l = [i for i in range(num)]
        random.shuffle(l)
        random_idx = torch.tensor(l)
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = (xo.unsqueeze(1) + xc[random_idx].unsqueeze(0)).contiguous().view(
                -1, self.args.hidden)
        '''

        if self.args.cat_or_add == "cat":
            xo.repeat_interleave(xo.shape[0], dim=0)
            xc = xc.repeat(xc.shape[0], 1)
            x = torch.cat((xc, xo), dim=1)
        else:
            # x = xc + xo
            x = (xo.unsqueeze(1) + xc.unsqueeze(0)).contiguous().view(
                -1, self.args.hidden)
        #  x=x.view(-1, self.args.hidden)
        
        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        x_logis = F.log_softmax(x, dim=-1)
        return x_logis


class GCNNet(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args, 
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0, 
                 edge_norm=True):
        super(GCNNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        hidden = args.hidden
        num_conv_layers=3

        hidden_in = num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))
            
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        x = self.lin_class(x)
        return x

class GINNet(torch.nn.Module):
    def __init__(self, num_features,
                       num_classes,
                       args, 
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0):

        super(GINNet, self).__init__()
        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden = args.hidden
        hidden_in = num_features
        hidden_out = num_classes
        
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(GINConv(
            Sequential(Linear(hidden, hidden), 
                       BatchNorm1d(hidden), 
                       ReLU(),
                       Linear(hidden, hidden), 
                       ReLU())))

        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        # x, edge_index, batch = data.feat, data.edge_index, data.batch
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))    
        x = self.bn_hidden(x)
        x = self.lin_class(x)

        return x

class GATNet(torch.nn.Module):
    def __init__(self, num_features, 
                       num_classes,
                       args,
                       head=4,
                       num_fc_layers=2, 
                       num_conv_layers=3, 
                       dropout=0.2):

        super(GATNet, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden = args.hidden
        hidden_in = num_features
        hidden_out = num_classes
   
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, hidden_out)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data):
        
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index))

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)

        return x


class CausalGIN(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       num_classes, args,
                gfn=False,
                edge_norm=True):
        super(CausalGIN, self).__init__()

        hidden = args.hidden
        num_conv_layers = args.layers
        self.args = args
        self.global_pool = global_add_pool
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)
        hidden_in = num_features
        self.num_classes = num_classes
        hidden_out = num_classes
        self.fc_num = args.fc_num
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        for i in range(num_conv_layers):
            self.convs.append(GINConv(
            Sequential(
                       Linear(hidden, hidden), 
                       BatchNorm1d(hidden), 
                       ReLU(),
                       Linear(hidden, hidden), 
                       ReLU())))

        self.edge_att_mlp = nn.Linear(hidden * 2, 2)
        self.node_att_mlp = nn.Linear(hidden, 2)
        self.bnc = BatchNorm1d(hidden)
        self.bno= BatchNorm1d(hidden)
        self.context_convs = GConv(hidden, hidden)
        self.objects_convs = GConv(hidden, hidden)

        # context mlp
        self.fc1_bn_c = BatchNorm1d(hidden)
        self.fc1_c = Linear(hidden, hidden)
        self.fc2_bn_c = BatchNorm1d(hidden)
        self.fc2_c = Linear(hidden, hidden_out)
        # object mlp
        self.fc1_bn_o = BatchNorm1d(hidden)
        self.fc1_o = Linear(hidden, hidden)
        self.fc2_bn_o = BatchNorm1d(hidden)
        self.fc2_o = Linear(hidden, hidden_out)
        # random mlp
        if self.args.cat_or_add == "cat":
            self.fc1_bn_co = BatchNorm1d(hidden * 2)
            self.fc1_co = Linear(hidden * 2, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)

        elif self.args.cat_or_add == "add":
            self.fc1_bn_co = BatchNorm1d(hidden)
            self.fc1_co = Linear(hidden, hidden)
            self.fc2_bn_co = BatchNorm1d(hidden)
            self.fc2_co = Linear(hidden, hidden_out)
        else:
            assert False
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, data, eval_random=True, train_type="base"):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
        
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        node_weight_c = node_att[:, 0]
        node_weight_o = node_att[:, 1]
        
        
        xc = node_weight_c.view(-1, 1) * x
        xo = node_weight_o.view(-1, 1) * x
        # xc = F.relu(self.context_convs(self.bnc(xc), edge_index, edge_weight_c))
        xc = F.relu(self.objects_convs(self.bnc(xc), edge_index, edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index, edge_weight_o))


        self.h_graph = self.global_pool(x, batch)
        xc = self.global_pool(xc, batch)
        xo = self.global_pool(xo, batch)

        self.h_mix_list = self.construct_h_mix_list(xc, xo)
        
        xc_logis = self.context_readout_layer(xc)
        xco_logis = self.random_readout_layer(xc, xo, eval_random=eval_random)
        # return xc_logis, xo_logis, xco_logis
        xo_logis = self.objects_readout_layer(xo, train_type)
        return xc_logis, xo_logis, xco_logis
        
    def construct_h_mix_list(self, xc, xo):
        """
        imput:
        xo: Causal embedding, shape = (batch_size, embedding_dim)
        xc: Bias embedding, shape = (batch_size, embedding_dim)

        return:
        h_mix_list: shape = (batch_size, batch_size - 1, embedding_dim)
        """
        self.batch_size = xc.size(0)
        idx = torch.arange(self.batch_size, device=xc.device)
        causal_idx = idx.unsqueeze(1).repeat(1, self.batch_size - 1).flatten()
        bias_idx = idx.unsqueeze(1).repeat(1, self.batch_size).T

        mask = ~torch.eye(self.batch_size, dtype=torch.bool, device=xc.device)
        bias_idx = bias_idx[mask]

        h_mix_list = xo[causal_idx] + xc[bias_idx]
        h_mix_list = h_mix_list.view(self.batch_size, self.batch_size - 1, -1) # (batch_size, batch_size - 1, embedding_dim)

        return h_mix_list

    def context_readout_layer(self, x):
        
        x = self.fc1_bn_c(x)
        x = self.fc1_c(x)
        x = F.relu(x)
        x = self.fc2_bn_c(x)
        x = self.fc2_c(x)
        return x
        # x_logis = F.log_softmax(x, dim=-1)
        # return x_logis

    def objects_readout_layer(self, x, train_type):
   
        x = self.fc1_bn_o(x)
        x = self.fc1_o(x)
        x = F.relu(x)
        x = self.fc2_bn_o(x)
        x = self.fc2_o(x)
        return x
        # x_logis = F.log_softmax(x, dim=-1)
        # if train_type == "irm":
            # return x, x_logis
        # else:
            # return x_logis

    def random_readout_layer(self, xc, xo, eval_random):

        num = xc.shape[0]
        l = [i for i in range(num)]
        if eval_random:
            random.shuffle(l)
        random_idx = torch.tensor(l)
        
        if self.args.cat_or_add == "cat":
            x = torch.cat((xc[random_idx], xo), dim=1)
        else:
            x = xc[random_idx] + xo

        x = self.fc1_bn_co(x)
        x = self.fc1_co(x)
        x = F.relu(x)
        x = self.fc2_bn_co(x)
        x = self.fc2_co(x)
        return x
        # x_logis = F.log_softmax(x, dim=-1)
        # return x_logis
    
    def eval_forward(self, data, train_type="base"):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        row, col = edge_index
        x = self.bn_feat(x.float())
        x = F.relu(self.conv_feat(x, edge_index))

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_att = F.softmax(self.edge_att_mlp(edge_rep), dim=-1)
        #  edge_weight_c = edge_att[:, 0]
        edge_weight_o = edge_att[:, 1]

        node_att = F.softmax(self.node_att_mlp(x), dim=-1)
        #  node_weight_c = node_att[:, 0]
        node_weight_o = node_att[:, 1]

        #   xc = node_weight_c.view(-1, 1) * x
        xo = node_weight_o.view(-1, 1) * x
        # xc = F.relu(self.context_convs(self.bnc(xc), edge_index,
        #                                edge_weight_c))
        xo = F.relu(self.objects_convs(self.bno(xo), edge_index,
                                       edge_weight_o))

        #    xc = self.global_pool(xc, batch)#short cut feature(batch_size*hidden)
        xo = self.global_pool(xo, batch)  #causal feature (batch_size*hidden)

        xo_logis = self.objects_readout_layer(xo, train_type)
        return xo_logis

    def similarity(self, h1, h2):
        h1 = F.normalize(h1, p=2, dim=-1)
        h2 = F.normalize(h2, p=2, dim=-1)
        return torch.matmul(h1, h2.T)

    # Loss function for mixed graphs (L_mix)
    def mix_loss(self):
        """
        h_mix_list: List of Tensors of shape (B, D), each for a different bias environment.
        h_graph: Tensor of shape (B, D), original graph embeddings.
        batch_size: Number of graphs in a batch.
        K: Number of bias environments.
        """
        K = self.batch_size -1
        loss = 0
        for i in range(self.batch_size):
            for j in range(K):
                # Positive sample: h_mix_list[j][i] and h_graph[i]
                # print(self.h_mix_list[i][j].shape)
                # print(self.h_graph[i].shape)
                pos_sim = self.similarity(self.h_mix_list[i][j], self.h_graph[i])

                # Negative samples: h_mix_list[j][i] and all other h_graph[k] (k != i)
                neg_sims = self.similarity(self.h_mix_list[i][j], self.h_graph)
                neg_sims[i] = -float('inf')  # Mask self similarity

                # numerator = torch.exp(pos_sim)
                # denominator = torch.exp(pos_sim) + torch.exp(neg_sims).sum()
                # loss += -torch.log(numerator / denominator)
                # InfoNCE loss for this environment
                loss += -torch.log(torch.exp(pos_sim) / torch.exp(neg_sims).sum())

        return loss / (self.batch_size * K)

    # Loss function for environment consistency (L_env)
    def env_loss(self):
        """
        h_mix_list: List of Tensors of shape (B, D), each for a different bias environment.
        batch_size: Number of graphs in a batch.
        K: Number of bias environments.
        """
        K = self.batch_size -1
        loss = 0
        for i in range(self.batch_size):
            for j in range(K):
                for l in range(K):
                    if j != l:
                        loss += self.similarity(self.h_mix_list[i][j], self.h_mix_list[i][l])

        return loss / (self.batch_size * K * (K - 1))