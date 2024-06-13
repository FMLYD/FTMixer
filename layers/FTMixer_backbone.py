__all__ = ['FTMixer_backbone']

import math
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import torch_dct as dct
from einops import rearrange
# from collections import OrderedDict
from layers.FTMixer_layers import *
from layers.RevIN import RevIN
from mamba_ssm import Mamba
# Cell
 
class Variable(nn.Module):
    def __init__(self,context_window,target_window,m_layers,d_model,dropout,c_in):
        super(Variable,self).__init__()
        self.mambas=nn.ModuleList([Mamba(d_model=d_model,  # Model dimension d_model
            d_state=2,  # SSM state expansion factor
            d_conv=2,  # Local convolution width
            expand=1,  # Block expansion factor)
            )for _ in range(m_layers)])
        self.convs=nn.ModuleList([nn.Sequential(nn.Linear(d_model,d_model))for _ in range(m_layers)])
        self.pwconvs=nn.ModuleList([nn.Sequential(nn.Conv1d(c_in,c_in,1,1))for _ in range(m_layers)])

        self.layers=m_layers
        self.up=nn.Linear(context_window,d_model)
        self.down=nn.Linear(d_model,target_window)
        self.bns=nn.ModuleList([nn.LayerNorm(d_model)for _ in range(m_layers)])
        self.bnv=nn.ModuleList([nn.BatchNorm1d(c_in)for _ in range(m_layers)])

        self.act=nn.SELU()
        self.dropout=nn.Dropout(dropout)
        self.Linears=nn.ModuleList([nn.Sequential(nn.Linear(d_model,d_model*2),nn.SELU(),nn.Linear(d_model*2,d_model),nn.LayerNorm(d_model))for _ in range(m_layers)])
    
    def forward(self,x):
        x=dct.dct(x)
        for i in range(self.layers):
            if i==0:
                x=self.up(x)
            x=self.convs[i](x)
            x=self.dropout(x)+x
            x=self.bns[i](x)
            x=self.pwconvs[i](x)
            x=self.dropout(x)+x
            x=self.bnv[i](x)
            if i==self.layers-1:
                x=self.down(x)
                x=dct.idct(x)
        return x if self.layers >0 else 0
class FTMixer_backbone(nn.Module):
    def get_para(self):
        weights=self.linear.weight.data.detach().cpu()
        # weights=F.softmax(weights,dim=0)
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        import seaborn as sns
        import matplotlib.pyplot as plt
        cmap = LinearSegmentedColormap.from_list(
    'custom_cmap',
    [(0, 'blue'), (0.5, 'white'), (1, 'red')]
)

        ax = sns.heatmap(weights,cmap=cmap,center=0, linewidth=0)
        plt.savefig('time.pdf',format='pdf')

    def __init__(self, c_in: int, context_window: int, target_window: int,
                 period, patch_len, stride, kernel_list, serial_conv=False, wo_conv=False, add=False,
                 max_seq_len: Optional[int] = 1024,m_model=512,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None,
                 d_v: Optional[int] = None,v_dropout=0.9,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = False,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False,
                 verbose: bool = False,m_layers=1, **kwargs):
        super().__init__()
        self.n=3
        #self.skip=nn.Linear(context_window,target_window)
        self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.period_list = period
        self.period_len = [math.ceil(context_window / i) for i in self.period_list]
        self.kernel_list = [(n, patch_len[i]) for i, n in enumerate(self.period_len)]
        self.stride_list = [(n , m // 2 if stride is None else stride[i]) for i, (n, m) in enumerate(self.kernel_list)]
        self.d_model=d_model
        self.cin=c_in
        self.dim_list = [ k[0] * k[1] for k in self.kernel_list]
        self.tokens_list = [
            (self.period_len[i] // s[0]) *
            ((math.ceil(self.period_list[i] / k[1]) * k[1] - k[1]) // s[1] + 1)
            for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ]
        self.var=Variable(context_window,target_window,m_layers,m_model,v_dropout,c_in)
        self.pad_layer = nn.ModuleList([nn.ModuleList([
            nn.ConstantPad1d((0, p-context_window%p), 0)if context_window % p != 0 else nn.Identity(),
            nn.ConstantPad1d((0, k[1] - p % k[1]), 0) if p % k[1] != 0 else nn.Identity()
        ]) for p, (k, s) in zip(self.period_list, zip(self.kernel_list, self.stride_list))
        ])

        self.embedding = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, self.dim_list[i], kernel_size=k, stride=s),
            nn.Flatten(start_dim=2)
        ) for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ])
        self.embedding1 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, self.dim_list[i], kernel_size=k, stride=s),
            nn.Flatten(start_dim=2)
        ) for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ])
        self.backbone =nn.Sequential( TSTiEncoder(c_in, patch_num=sum(self.tokens_list), patch_len=1, max_seq_len=max_seq_len,
                        n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                        norm=norm, attn_dropout=attn_dropout, dropout=dropout, act=act,
                        key_padding_mask=key_padding_mask, padding_var=padding_var,
                        attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                        store_attn=store_attn,individual=individual,
                        pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs),nn.Flatten(start_dim=-2),nn.Linear(sum(self.tokens_list)*d_model,context_window)
            )
        self.clinear1=nn.Linear(target_window,target_window*10).to(torch.cfloat)
        self.last=nn.Linear(context_window,target_window)
        self.wo_conv = wo_conv
        self.serial_conv = serial_conv
        self.compensate=(context_window+target_window)/context_window
        if not self.wo_conv:
            self.conv = nn.Sequential(*[
                nn.Sequential(nn.Conv1d(self.n+1, self.n+1, kernel_size=i, groups=self.n+1, padding='same'), nn.SELU(),nn.Dropout(fc_dropout),nn.BatchNorm1d(self.n+1))
                for i in kernel_list],
                nn.Flatten(start_dim=-2),
                nn.Linear(context_window*(self.n+1),context_window)
            )
            
  
            self.conv1 = nn.ModuleList([nn.Sequential(*[
                nn.Sequential(nn.Conv1d(n, n, kernel_size=i, groups=n, padding='same'), nn.SELU(),nn.BatchNorm1d(n))
                for i in kernel_list],
                nn.Dropout(fc_dropout),
            ) for n in self.period_len])
        self.dual=nn.Linear(context_window,target_window)
        self.conv_drop=nn.Dropout(fc_dropout)
        self.glo=nn.ModuleList([nn.Linear(context_window,context_window) for i in range(len(period))])
        self.proj=nn.ModuleList([nn.Linear(context_window,context_window) for _ in range(len(period))])
       # self.mamba = Variable(context_window,m_model
        #,m_layers,dropout)

        self.linear=nn.Linear(context_window,target_window)
        

        self.individual=individual
        if individual==False:
            self.W_P=nn.ModuleList([nn.Linear(self.dim_list[i],d_model)for i in range(len(self.dim_list))])
            self.W_P1=nn.ModuleList([nn.Linear(self.dim_list[i],d_model)for i in range(len(self.dim_list))])

        else:
            self.W_P1=nn.ModuleList([nn.Linear(self.dim_list[i],d_model)for i in range(len(self.dim_list))])
            self.loc_W_p1=nn.ModuleList([nn.ModuleList([nn.Linear(self.dim_list[i],d_model) for _ in range(c_in)]) for i in range(len(self.dim_list))])

            self.W_P=nn.ModuleList([nn.Linear(self.dim_list[i],d_model)for i in range(len(self.dim_list))])
            self.loc_W_p=nn.ModuleList([nn.ModuleList([nn.Linear(self.dim_list[i],d_model) for _ in range(c_in)]) for i in range(len(self.dim_list))])
        
        self.head = Head(context_window, 1, target_window, head_dropout=head_dropout, Concat=not add)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
        self.bn=nn.ModuleList([nn.BatchNorm1d(self.period_len[i])  for i in range(len(self.period_len)) ])
    def decouple(self,z,linear_all,linears,n):
        store=[]
        def sub_decouple(z,linears,n,store):
            if n==0:return 
            n=n-1
            index_tensor = torch.arange(z.size(-1))
            odd_indices = index_tensor % 2 != 0
            z_odd=z[:,:,odd_indices]
            z_even=z[:,:,~odd_indices]
            
            sub_decouple(z_odd,linears,n,store)
            sub_decouple(z_even,linears,n,store)
            #z_odd=dct.dct(z_odd)
            #z_even=dct.dct(z_even)
            z1=self.linears[n](z_odd)+self.linears[n](z_even)
            #z1=z_odd-z_even
            try:
            #z1=linears[n](z1)
                pass
            except:
                print(n)
                exit()
            store.append(z1)
            if n==0:return
        sub_decouple(z,linears,n,store)
        res=torch.cat(store,dim=-1)
        #res=F.leaky_relu(res)
        res=linear_all(res)
        return res
    def decouple1(self,z,n):
        def sub_decouple(z,n):
            if n==0:return None
            n=n-1
            index_tensor = torch.arange(z.size(-1))
            odd_indices = index_tensor % 2 != 0
            z_odd=z[:,:,odd_indices]
            z_even=z[:,:,~odd_indices]

            tmp1=sub_decouple(z_odd,n)
            if tmp1==None:
                #z_odd=dct.dct(z_odd)
                #z_even=dct.dct(z_even)

                z1=dct.dct(torch.cat([z_odd,z_even],dim=-1)).unsqueeze(-1)
                return z1
            tmp2=sub_decouple(z_even,n)
            z_odd=dct.dct(z_odd)
            z_even=dct.dct(z_even)
            z1=dct.dct(torch.cat([z_odd,z_even],dim=-1)).unsqueeze(-1)
            tmp=torch.cat([tmp1,tmp2],dim=-2)
            z1=torch.cat([z1,tmp],dim=-1)
            #z1=self.linears[n](z_odd)+self.linears[n](z_even)
            #z1=z_odd-z_even
            try:
            #z1=linears[n](z1)
                pass
            except:
                print(n)
                exit()

            return z1
        z1=sub_decouple(z,n)
        res=torch.cat([dct.dct(z).unsqueeze(-1),z1],dim=-1)
        #res=torch.cat(store,dim=-1)
        #res=F.leaky_relu(res)
        #res=linear_all(res)
        return res
    def forward(self, z):  # z: [bs x nvars x seq_len]

        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, 'norm')
        z = z.permute(0, 2, 1)
        res = []
        #loc1=dct.idct((self.mamba(dct.dct(z))))#.reshape(z.shape[0] * z.shape[1], -1, period)
        #loc1=self.var_down(loc1)
        skip=self.var(z)
        for i, period in enumerate(self.period_list):
            

            loc=((dct.dct(self.pad_layer[i][0](z).reshape(z.shape[0] * z.shape[1], -1, period))))#.permute(0,2,1)
            loc =((dct.idct((self.conv1[i](loc)))))#.reshape(z.shape[0], z.shape[1], -1)[..., :z.shape[-1]]
            x = self.pad_layer[i][0](z).reshape(z.shape[0] * z.shape[1], -1, period)+loc
            
            glo=x #+loc*F.sigmoid(x)#+loc*F.sigmoid(x)
            glo = self.pad_layer[i][1](glo)
            loc=self.pad_layer[i][1](loc)
            loc=loc.unsqueeze(-3)
            glo=glo.unsqueeze(-3)
            glo = self.embedding[i](glo)
            loc=self.embedding1[i](loc)
            
            glo = rearrange(glo, "(b m) d n -> b m d n", b=z.shape[0]).contiguous()
            glo=glo.permute(0,1,3,2)
            loc = rearrange(loc, "(b m) d n -> b m d n", b=z.shape[0]).contiguous()
            loc=loc.permute(0,1,3,2)
            if not self.individual:
                glo = self.W_P[i](glo)  # x: [bs x nvars x patch_num x d_model]
            else:
                tmp=[]
                tmp=torch.zeros((glo.shape[0],glo.shape[1],glo.shape[2],self.d_model)).to(glo.dtype).to(glo.device)
                for j in range(self.cin):
                    
                    tmp[:,i,:,:]=self.loc_W_p[i][j](glo[:,i,:,:])
                glo=self.W_P[i](glo)+tmp
            if not self.individual:
                loc = self.W_P1[i](loc)  # x: [bs x nvars x patch_num x d_model]
            else:
                tmp=[]
                tmp=torch.zeros((glo.shape[0],glo.shape[1],glo.shape[2],self.d_model)).to(glo.dtype).to(glo.device)
                for j in range(self.cin):
                    tmp[:,i,:,:]=self.loc_W_p1[i][j](loc[:,i,:,:])
                loc=self.W_P1[i](loc)+tmp
            # glo=glo+loc
            glo=glo.permute(0,1,3,2)
            glo=glo#+dct.idct(self.var_down(dct.dct(z))).unsqueeze(-1)
            res.append(glo)
        glo=torch.cat(res,dim=-1)
        z=self.linear(self.backbone(glo))+skip
        #z=F.sigmoid(skip)*z+F.sigmoid(z)*skip#*self.compensate#+skip
        #z=dct.idct(z)
    #+loc1
        #z=z.to(torch.cfloat)
        #z=self.clinear1(z)
        #z=torch.fft.ifft(z,dim=-1).float()
        #*self.compensate
        # z = self.last(glo)
        

        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, 'denorm')
        z = z.permute(0, 2, 1)
        return z

class Head(nn.Module):
    def __init__(self, context_window, num_period, target_window, head_dropout=0,
                 Concat=True):
        super().__init__()
        self.Concat = Concat
        self.linear = nn.Linear(context_window * (num_period if Concat else 1), target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.Concat:
            x = torch.cat(x, dim=-1)
            x = self.linear(x)
        else:
            x = torch.stack(x, dim=-1)
            x = torch.mean(x, dim=-1)
            x = self.linear(x)
        x = self.dropout(x)
        return x

class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=False, pre_norm=False,
                 pe='zeros',individual=False, learn_pe=True, verbose=False, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len
        res_attention=False
        # Input encoding
        q_len = patch_num
        if individual==False:
            self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        else:
            self.W_P=nn.Linear(patch_len,d_model)
            self.loc_W_p=nn.ModuleList([nn.Linear(patch_len,d_model) for _ in range(c_in)])
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.indivi=individual
        self.cin=c_in
        # Encoder
        self.d_model=d_model
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn, pos=self.W_pos)
        
    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        #x=dct.dct(x)
        
        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
        #z=dct.idct(z)
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z

    # Cell


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False,
                 pos=None
                 ):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn, pos=pos) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        
        for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False, pos=None):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        
        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                nn.SELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))
        
        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        self.dw_conv=nn.Conv1d(d_model,d_model,kernel_size=27,stride=1,padding='same',groups=d_model)
        self.conv1=nn.Linear(d_model,d_model)
        self.conv2=nn.Linear(d_model,d_model)
        
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.activation=nn.SELU()
    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        # if self.res_attention:
        #     src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
        #                                         attn_mask=attn_mask)
        # else:
        #     src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # if self.store_attn:
        #     self.attn = attn
        #
        #src2=self.mamba(src)
        #src=dct.dct(src)
        src2=self.dw_conv(src.permute(0,2,1)).permute(0,2,1)
        src2=self.activation(src2)
        ## Add & Norm
        src2 = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        src=src2#*F.sigmoid(self.conv1(src))
        if not self.pre_norm:
            src = self.norm_attn(src)
        
        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src2 = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        src=src2#*F.sigmoid(self.conv2(src))
        if not self.pre_norm:
            src = self.norm_ffn(src)
        return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False, pos=None):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.pos = pos
        self.P_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.P_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                   res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        q_p = self.P_Q(self.pos).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_p = self.P_K(self.pos).view(1, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                                              q_p=q_p, k_p=k_p)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor((head_dim * 1) ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None,
                q_p=None, k_p=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]
        # attn_scores += torch.matmul(q_p, k) * self.scale
        # attn_scores += torch.matmul(q, k_p) * self.scale

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        
        return output, attn_weights
