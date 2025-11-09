# -*- coding:utf-8 -*-
from re import T
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F


class conv2d_(nn.Module):
    def __init__(self, input_dims, output_dims, kernel_size, stride=(1, 1),
                 padding='SAME', use_bias=True, activation=F.relu,
                 bn_decay=None):
        super(conv2d_, self).__init__()
        self.activation = activation
        self.bn_decay = bn_decay
        if padding == 'SAME':
            self.padding_size = math.ceil(kernel_size)
        else:
            self.padding_size = [0, 0]
        self.conv = nn.Conv2d(input_dims, output_dims, kernel_size, stride=stride,
                              padding=0, bias=use_bias)
        decay = self.bn_decay if self.bn_decay is not None else 0.1
        self.batch_norm = nn.BatchNorm2d(output_dims, eps=1e-3, momentum=decay)
        if self.activation is not None:
            self.relu = nn.ReLU()
        
        self.initialize_weights(use_bias)
    
    def initialize_weights(self, use_bias=True):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_uniform_(self.conv.weight)
                # nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if use_bias:
                    nn.init.constant_(self.conv.bias, 0)
            if isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
                # nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')


    def forward(self, x):
        # pytorch conv2: B,C,H,W
        x = x.permute(0, 3, 2, 1)
        x = F.pad(x, ([self.padding_size[1], self.padding_size[1], self.padding_size[0], self.padding_size[0]]))
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation is not None:
            x = self.relu(x)

        return x.permute(0, 3, 2, 1)


class FC(nn.Module):
    def __init__(self, input_dims, units, activations, bn_decay, use_bias=True):
        super(FC, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        elif isinstance(units, tuple):
            units = list(units)
            input_dims = list(input_dims)
            activations = list(activations)
        assert type(units) == list
        self.convs = nn.ModuleList([conv2d_(
            input_dims=input_dim, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn_decay=bn_decay) for input_dim, num_unit, activation in
            zip(input_dims, units, activations)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class gatedFusion(nn.Module):
    '''
    gated fusion
    HS:     [batch_size, num_step, num_vertex, D]
    HT:     [batch_size, num_step, num_vertex, D]
    D:      output dims
    return: [batch_size, num_step, num_vertex, D]
    '''

    def __init__(self, D, bn_decay):
        super(gatedFusion, self).__init__()
        self.FC_xs = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=False)
        self.FC_xt = FC(input_dims=D, units=D, activations=None,
                        bn_decay=bn_decay, use_bias=True)
        self.FC_h = FC(input_dims=[D, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self.FC_h(H)
        del XS, XT, z
        return H


class STEmbedding(nn.Module):
    '''
    spatio-temporal embedding
    SE:     [num_vertex, D]
    TE:     [batch_size, num_his + num_pred, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, num_his + num_pred, num_vertex, D]
    '''

    def __init__(self, D, bn_decay=0.1):
        super(STEmbedding, self).__init__()
        self.FC_se = FC(
            input_dims=[4096, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)

        self.FC_te = FC(
            input_dims=[2, D], units=[D, D], activations=[F.relu, None],
            bn_decay=bn_decay)  # time of day sin cos

    def forward(self, SE, TE):
        # spatial embedding
        SE = SE.unsqueeze(0).unsqueeze(0)
        # print("SE.shape = ", SE.shape)
        SE = self.FC_se(SE)
        TE = self.FC_te(TE)
        
        return torch.add(SE, TE)
        
class SpatialGroupingModule(torch.nn.Module):
    def __init__(self, dim_in, dim_out, num_sg, num_cg, input_length):
        super(SpatialGroupingModule, self).__init__()
        assert dim_in % num_cg == 0
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_sg = num_sg    # spatial group
        self.num_cg = num_cg    # channel group (#TODO)
        self.tau = 1
        self.t_cov = torch.nn.Sequential(
                  torch.nn.Conv2d(dim_in*input_length, dim_out, kernel_size=3, padding=1),
                  torch.nn.ReLU(),
                  torch.nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1),
                  )
        self.wc = torch.nn.Sequential(
                  torch.nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
                  torch.nn.ReLU(),
                  torch.nn.Conv2d(dim_out, num_sg * num_cg * num_cg, kernel_size=3, padding=1),
                  )

    def forward(self, x):
        # postfix ``s'' indicates spatial, ``c'' indicates center
        # get spatial features; x = B T N D
        x = x.permute(0, 2, 1, 3)                                                       # b x N x T x D
        # print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1], -1)                                       # b x N x TD
        x = x.unsqueeze(1)                                                              # b x 1 x N x TD
        x = x.permute(0, 3, 2, 1)                                                       # b x TD x N x 1
        # print(x.shape)
        x = self.t_cov(x)                                                               # b x D x N x 1
        zs = x.transpose(1,2)                                                           # b x N x D x 1
        zs = zs.reshape(zs.shape[0], zs.shape[1], -1)                                      # b x N x D
        zs = zs.reshape(zs.shape[0], zs.shape[1], self.num_cg, self.dim_in//self.num_cg)   # b x N x cg x D//cg
        zs = zs.reshape(zs.shape[0], zs.shape[1]*self.num_cg, self.dim_in//self.num_cg)    # b x N*cg x D//cg
        
        # get weights for centers1
        # wc = x.view(-1, x.shape[2], x.shape[3]).transpose(1,2)                          # b x D x N
        # wc = wc.unsqueeze(-1)                                                           # b x D x N x 1
        wc = self.wc(x)                                                                 # b x sg*cg*cg x N x 1
        wc = wc.reshape(wc.shape[0], wc.shape[1], -1)                                      # b x sg*cg*cg x N
        wc = wc.reshape(wc.shape[0], wc.shape[1]//self.num_cg, self.num_cg*wc.shape[-1])   # b x sg*cg x cg*N
        wc = torch.softmax(wc, dim=2)
        
        # get centers
        c = torch.bmm(wc, zs)                                                           # b x sg*cg x D//cg
        # print(torch.cdist(c, c))  # 检查中心点之间的距离
        # get weights for spatial features
        ws = torch.cdist(zs, c)                                                         # b x N*cg x sg*cg
        # print(ws.shape)
        
        ws = torch.softmax(-ws/self.tau, dim=2)                                         # b x N*cg x sg*cg
        # print(torch.mean(ws,dim=-1)[0], torch.std(ws, dim=-1)[0])
        ws_sum = ws.sum(dim=1, keepdim=True)
        ws = ws / (ws_sum.detach()+1e-6)
        # print(torch.min(ws,dim=-1)[0], torch.max(ws,dim=-1)[0], torch.sum(ws,dim=-1), torch.sum(ws, dim=-2))
        zc = torch.bmm(ws.transpose(1,2), zs)                                           # b x sg*cg x D//cg
        # print("zc ====")
        # print(torch.cdist(zc, zc))  # 检查中心点之间的距离
        
        return zc, ws



class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = torch.mean(kernels[:batch_size, :batch_size])
        YY = torch.mean(kernels[batch_size:, batch_size:])
        XY = torch.mean(kernels[:batch_size, batch_size:])
        YX = torch.mean(kernels[batch_size:, :batch_size])
        loss = torch.mean(XX + YY - XY -YX)
        return loss
    
class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        '''
        graph convolutional operation, a simple GCN we defined in STFGNN

        Parameters
        ----------
        A: (N, N)

        X:(B, N, C)

        in_features: int, C

        out_features: int, C'

        Returns
        ----------
        output shape is (B, N, C')
        '''
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stvd = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stvd,stvd)
        if self.bias is not None:
            self.bias.data.uniform_(-stvd,stvd)

    def forward(self, X, A):
        support = torch.matmul(X,self.weight)
        output = torch.matmul(A,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        

class GCNL(nn.Module):
    def __init__(self, number_of_filter, number_of_feature):
        '''
        graph convolutional operation, a simple GCN we defined in STFGNN

        Parameters
        ----------
        A: (N, N)

        X:(B, N, C)

        num_of_filter: int, C'

        num_of_features: int, C

        Returns
        ----------
        output shape is (B, N, C')
        '''
        super(GCNL, self).__init__()
        self.gcn1 = GCN(number_of_feature,number_of_filter,True)
        self.gcn2 = GCN(number_of_feature,number_of_filter,True)
        self.sig = nn.Sigmoid()

    def forward(self, X, A):
        lhs = self.gcn1(X, A)
        rhs = self.gcn2(X, A)
        return (lhs * self.sig(rhs))

    
class GCNBlock_Single(nn.Module):
    def __init__(self, num_of_feature, filters):
        '''
        graph convolutional operation, a simple GCN we defined in STFGNN

        Parameters
        ----------
        A: (N, N)

        num_of_node: int, N

        num_of_feature: int, C

        filters: list[int], list of C'

        X:(B, N, C)

        Returns
        ----------
        output shape is (B, N, C')
        '''
        super(GCNBlock_Single, self).__init__()
        self.gcn_layers = nn.ModuleList()
        # self.num_of_node = num_of_node
        for i in range(len(filters)):
            self.gcn_layers.append(GCNL(filters[i], num_of_feature))
            num_of_feature = filters[i]
            

    def forward(self, X, A):
        lay_output = []
        for gcn in self.gcn_layers:
            X = gcn(X, A)
            # shape (B, N, C')
            lay_output.append(X)
        # shape (L, B, N, C')
        lay_output = torch.stack(lay_output)
        # shape (B, N, C')
        z = torch.max(lay_output, dim=0)[0]
        return z
    
class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()
    

class USPBlock_Inductive_MSconv(nn.Module):

    def __init__(self, input_length, number_of_feature, filters, sg_num, cg_num):
        '''
        Combine the GNN and CNN together

        Parameters
        ----------

        input_length: int, length of time series, T

        num_of_node: int, N

        num_of_feature: int, C

        filters: list[int], list of C'

        X:(B, T, N, C)

        Returns
        ----------
        output shape is (B, N, T')
        '''

        super(USPBlock_Inductive_MSconv, self).__init__()
        self.input_length = input_length
        self.number_of_feature = number_of_feature
        self.sg_num = sg_num
        self.cg_num = cg_num

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        kernel_size = 2
        conv_channels = [64,32,64]
        num_level = len(conv_channels)
        layers = []
        for i in range(num_level):
            dilation_size = 2**i
            in_channels = number_of_feature if i == 0 else conv_channels[i-1]
            out_channels = conv_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tempnet = nn.Sequential(*layers)
        
        
        self.ts_windows = nn.ModuleList()
        for i in range(input_length):
            # input (B, N, C) output (B, N, C')
            self.ts_windows.append(GCNBlock_Single(number_of_feature, filters))

        self.tt_windows = nn.ModuleList()
        for i in range(input_length):
            # input (B, N, C) output (B, N, C')
            self.tt_windows.append(GCNBlock_Single(number_of_feature, filters))
        
        self.spa_gr = SpatialGroupingModule(number_of_feature, number_of_feature, sg_num, cg_num, input_length)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.414)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self, X, A_s, A_t):
        # CNN
        X_temp = X.clone()
        X_temp = X_temp.permute(0, 3, 2, 1)
        X_temp = self.tempnet(X_temp)
        x_res = X_temp.permute(0, 3, 2, 1)

        # GCN for A_S
        ts_slide = []
        for i in range(self.input_length):
            # input (B, N, C') output (B, N, C')
            zs = torch.reshape(X[:,i:i+1,:,:],(-1,X.shape[2],self.number_of_feature))
            # print(zs.shape)
            zs = self.ts_windows[i](zs,A_s)
            # print(zs.shape)
            ts_slide.append(torch.unsqueeze(zs, 1))
        # (B, T, N, C)
        x_sp = torch.cat(ts_slide, dim=1)

        # GCN for A_t
        tt_slide = []
        for i in range(self.input_length):
            # input (B, N, C') output (B, N, C')
            zt = torch.reshape(X[:,i:i+1,:,:],(-1,X.shape[2],self.number_of_feature))
            zt = self.tt_windows[i](zt,A_t)
            tt_slide.append(torch.unsqueeze(zt, 1))
        # (B, T, N, C)
        x_tp = torch.cat(tt_slide, dim=1)
        x_gcn = torch.stack([x_sp,x_tp])
        # (B, T-1, N, C')
        x_gcn = torch.max(x_gcn, dim=0)[0]
        # print("x_gcn.shape = ", x_gcn.shape)
        
        # adding spatial group module here
        

        # (B, T, N, C)
        output = x_gcn+x_res
        zc, attn = self.spa_gr(output)

        return output, attn, zc

# has dynamic temporal matrix
class USPGCN_MultiD_Inductive_Tattr(nn.Module):

    def __init__(self, input_length, number_of_feature, filters, device, output_length, sg_num, cg_num):
        '''
        graph convolutional operation, a simple GCN we defined in STFGNN

        Parameters
        ----------
        A: (N, N)

        T: int, length of time series, T

        num_of_node: int, N

        num_of_feature: int, C

        filters_list: list[list[int]], list of C'

        X:(B, T, N, C)

        Returns
        ----------
        output shape is (B, N, T')
        '''

        super(USPGCN_MultiD_Inductive_Tattr, self).__init__()
        self.device = device
        self.input_length = input_length
        self.number_of_feature = number_of_feature
        self.output_length = output_length

        self.fc_x = nn.Linear(1, number_of_feature)
        self.ste_emb = STEmbedding(number_of_feature)
        self.fc_te = nn.Linear(2, number_of_feature)
        self.fc_position = nn.Linear(2, number_of_feature)
        self.fc_xte = FC(input_dims=2 * number_of_feature, units=number_of_feature, activations=F.relu, bn_decay=0.1)
        self.fc_weather = nn.Linear(4, number_of_feature)
        num_heads = 8
        
        # Multi-head attention for cross-modal interaction
        self.cross_attention = nn.MultiheadAttention(embed_dim=number_of_feature, num_heads=num_heads, batch_first=True)
        
        self.traffic_out = nn.Sequential(
            nn.Linear(number_of_feature, number_of_feature),
            nn.ReLU(),
            nn.Linear(number_of_feature, number_of_feature)
        )
        
        self.weather_out = nn.Sequential(
            nn.Linear(number_of_feature, number_of_feature),
            nn.ReLU(),
            nn.Linear(number_of_feature, number_of_feature)
        )
        
        self.alpha_layer = nn.Linear(number_of_feature, 1)  # 输出维度为 T，表示每个时间步的权重

        self.gated_fusion = gatedFusion(number_of_feature, 0.1)
        
        # self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU(0.01)
        
        self.blocks = nn.ModuleList()
        for i in range(len(filters)):
            self.blocks.append(USPBlock_Inductive_MSconv(input_length, number_of_feature, filters[i], sg_num, cg_num))
            # input_length = input_length - 1

        self.fc_time = nn.Linear(input_length, output_length)
        self.fc_out = nn.Linear(filters[-1][-1], 1)
        self.project = nn.Sequential(
            nn.Linear(filters[-1][-1], filters[-1][-1]),
            nn.BatchNorm1d(filters[-1][-1]),
            nn.ReLU(),
            nn.Linear(filters[-1][-1], filters[-1][-1]),
            # nn.Sigmoid()
        )
        self.norm_layer = nn.LayerNorm(number_of_feature).to(device)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.414)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        with torch.no_grad():  
            if self.cross_attention.in_proj_weight is not None:
                q_weight, k_weight, v_weight = self.cross_attention.in_proj_weight.chunk(3, dim=0)
                nn.init.xavier_uniform_(q_weight)
                nn.init.xavier_uniform_(k_weight)
                nn.init.xavier_uniform_(v_weight)
                self.cross_attention.in_proj_weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))  # 重新赋值
            else:
                nn.init.xavier_uniform_(self.cross_attention.q_proj_weight)
                nn.init.xavier_uniform_(self.cross_attention.k_proj_weight)
                nn.init.xavier_uniform_(self.cross_attention.v_proj_weight)

        nn.init.xavier_uniform_(self.cross_attention.out_proj.weight)
        nn.init.constant_(self.cross_attention.out_proj.bias, 0)

    def forward(self, X, TE, position, weather, A_s, A_t):
        # X:(B, T, N, C') embedding X
        X = self.fc_x(X)
        B, T, N, D = X.shape
        # embedding position and TE
        STE = self.ste_emb(position, TE)
        X = self.fc_xte(torch.cat((X, STE), dim=-1))
        
        weather = self.fc_weather(weather)  
        
        LB = weather.shape[1]
        X = X.permute(0, 2, 1, 3).reshape(B * N, T, D)  # (B * N, T, hidden_dim)
        weather = weather.permute(0, 2, 1, 3).reshape(B * N, LB, D)  # (B * N, T, hidden_dim)
        
        X_weather, weather_att = self.cross_attention(query=X, key=weather, value=weather)
        X_weather = X_weather.reshape(B, N, T, D).permute(0, 2, 1, 3)
        X_weather = self.weather_out(X_weather)
        
        X = X.reshape(B, N, T, D).permute(0, 2, 1, 3)
        X = self.gated_fusion(X, X_weather)
        
        # (B, T, N, C)
        attn_list = []
        zc_list = []
        for block in self.blocks:
            X, attn, zc = block(X, A_s, A_t)
            attn_list.append(attn)
            zc_list.append(zc)
        
        rep = X.clone()
        rep = rep[:,-1:,:,:]
        rep = torch.squeeze(rep)
        # project head
        if rep.dim() == 3:
            rep = rep.transpose(1, 2)
        elif rep.dim() == 2:
            rep = rep.unsqueeze(0).transpose(1, 2)
        rep = torch.sum(rep, dim=2)
        rep = self.project(rep)  
        output = X.transpose(1,3)
        # split into two subsets
        output = self.leakyrelu(self.fc_time(output))
        output = output.transpose(1,3)
        output = self.leakyrelu(self.fc_out(output))
        
        return output, rep, attn_list, zc_list
