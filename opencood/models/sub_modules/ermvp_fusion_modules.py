"""
This class is about swap fusion applications
"""
import torch
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

from opencood.models.base_transformer import FeedForward, PreNormResidual
from torch.nn.functional import interpolate as interpolate

class SE(nn.Module):
    """
    Squeeze and excitation block
    """

    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    
class FeatExtract(nn.Module):
    """
    Feature extraction block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """
    def __init__(self, dim,strides,keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=strides, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x
    
class WindowAttentionGlobal(nn.Module):
    """
    Global window attention based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """
    def __init__(
            self,
            dim = 256,
            dim_head=32,
            dropout=0.,
            agent_size=2,
            window_size=7
    ):
        super().__init__()
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """
        assert (dim % dim_head) == 0, \
            'dimension should be divisible by dimension per head'
        self.head_dim = dim_head
        self.num_heads = dim // dim_head
        self.scale =  dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]
        self.agent_size = agent_size
        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) *
            (2 * self.window_size[1] - 1) *
            (2 * self.window_size[2] - 1),
            self.num_heads)  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # 3, Wd, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = \
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= \
            (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 2)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.to_q_global = nn.Sequential(
                FeatExtract(dim,(2,2),keep_dim=False),
                FeatExtract(dim,(2,2), keep_dim=False),
                # FeatExtract(dim, keep_dim=False),
            )
        # self.conv3d = nn.Conv3d(self.max_cav, 1, 1, stride=1, padding=0)
    def forward(self, x,mask=None):
        batch, agent_size, height, width, window_height, window_width, _, device, h \
            = *x.shape, x.device, self.num_heads
        # x.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(0, 1, 3, 2, 4)
        #[1，2，15，45，4，4，256]
        x_global = rearrange(x, 'b m x y w1 w2 d -> (b m) d (x w1) (y w2)')
   
        x_global = self.to_q_global(x_global)
        B, C, H, W = x_global.shape

        x_global = rearrange(x_global, '(b m) d (x w1) (y w2) -> b m x y w1 w2 d',
                      m=agent_size,w1=window_height, w2=window_height)
        x_global = rearrange(x_global, 'b l x y w1 w2 d -> b (x y) (l w1 w2) d')
        g_attn = (x_global @ x_global.transpose(-2, -1))
        g_attn = self.softmax(g_attn)*(C**-0.5)

        x_global = einsum('b h i j, b h j d -> b h i d', g_attn, x_global)

        q_global = torch.mean(x_global, dim=1, keepdim=True)
        q_global = q_global.reshape(batch,1,agent_size*window_height*window_height,self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        B_, N, C = x.shape
        B = q_global.shape[0]
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        B_dim = torch.div(B_, B, rounding_mode='floor')
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q_global = q_global.repeat(1, B_dim, 1, 1, 1)
        q = q_global.reshape(B_, self.num_heads, N, head_dim)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))


        relative_position_bias = self.relative_position_bias_table(self.relative_position_index)
        # sim = sim + rearrange(bias, 'i j h -> h i j')
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        # mask shape if exist: b x y w1 w2 e l
        if mask is not None:
            # [675,1,1,32]
            # b x y w1 w2 e l -> (b x y) 1 (l w1 w2)
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            # (b x y) 1 1 (l w1 w2) = b h 1 n
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -float('inf'))

        attn = self.softmax(attn)
        # attn = self.attn_drop(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)',
                        l=agent_size, w1=window_height, w2=window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d',
                         b=batch, x=height, y=width)

class Attention(nn.Module):
    """
    Unit Attention class. Todo: mask is not added yet.

    Parameters
    ----------
    dim: int
        Input feature dimension.
    dim_head: int
        The head dimension.
    dropout: float
        Dropout rate
    agent_size: int
        The agent can be different views, timestamps or vehicles.
    """

    def __init__(
            self,
            dim = 256,
            dim_head=32,
            dropout=0.,
            agent_size=6,
            window_size=7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, \
            'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(
            nn.Softmax(dim=-1)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) *
            (2 * self.window_size[1] - 1) *
            (2 * self.window_size[2] - 1),
            self.heads)  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # 3, Wd, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = \
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= \
            (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

    def forward(self, x, mask=None):
        # x shape: b, l, h, w, w_h, w_w, c
        # [1,2,15,45,4,4,256]
        batch, agent_size, height, width, window_height, window_width, _, device, h \
            = *x.shape, x.device, self.heads

        # flatten
        #[675(B*H*W/P^2),32,N*P^2,256 C]


        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        # project for queries, keys, values

        # self.to_qkv(x) ->    C->3C
        # chunk -1 表示沿着最后一个维度进行分割
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # split heads
        # [675,8,32,32]
        #'b' 仍然是批处理维度， 'h' 表示头（head）的数量， 'n' 表示序列长度维度， 'd' 表示每个头的维度
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      (q, k, v))
        # scale
        q = q * self.scale

        # sim
        # [675,8,32,32]
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        # [32,32,8]
        bias = self.relative_position_bias_table(self.relative_position_index)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # mask shape if exist: b x y w1 w2 e l
        if mask is not None:
            # [675,1,1,32]
            # b x y w1 w2 e l -> (b x y) 1 (l w1 w2)
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            # (b x y) 1 1 (l w1 w2) = b h 1 n
            mask = mask.unsqueeze(1)
            #使用 masked_fill 函数，将 sim 中与 mask 中的零值对应的位置的值替换为负无穷（-inf）。这意味着这些位置的注意力权重将被设置为零，即模型不会关注这些位置。
            sim = sim.masked_fill(mask == 0, -float('inf'))

        # attention  Softmax函数
        attn = self.attend(sim)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)',
                        l=agent_size, w1=window_height, w2=window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d',
                         b=batch, x=height, y=width)

class ERMVPFusionBlockMask(nn.Module):

    def __init__(self,
                 input_dim,
                 mlp_dim,
                 dim_head,
                 window_size,
                 agent_size,
                 drop_out):
        super(ERMVPFusionBlockMask, self).__init__()

        self.window_size = window_size

        self.window_attention = PreNormResidual(input_dim,
                                                Attention(input_dim, dim_head,
                                                          drop_out,
                                                          agent_size,
                                                          window_size))
        self.window_ffd = PreNormResidual(input_dim,
                                          FeedForward(input_dim, mlp_dim,
                                                      drop_out))
        self.grid_attention = PreNormResidual(input_dim,
                                              Attention(input_dim, dim_head,
                                                        drop_out,
                                                        agent_size,
                                                        window_size))
        self.WindowAttentionGlobal = PreNormResidual(input_dim,
                                              WindowAttentionGlobal(input_dim, dim_head,
                                                        drop_out,
                                                        agent_size,
                                                        window_size))
        self.windowGlobal_ffd = PreNormResidual(input_dim,
                                          FeedForward(input_dim, mlp_dim,
                                                      drop_out))
        self.grid_ffd = PreNormResidual(input_dim,
                                        FeedForward(input_dim, mlp_dim,
                                                    drop_out))
        # self.TokenInitializer = TokenInitializer(input_dim,window_size)
        


    def forward(self, x, mask):
        # x: b l c h w
        # mask: b h w 1 l
        # window attention -> grid attention
        # [1,60,180,1,2]

                # [1,60,180,1,2]
        
        mask_swap = mask

        # mask b h w 1 l -> b x y w1 w2 1 L
        #   [1,15,45,4,4,1,2]
        mask_swap = rearrange(mask_swap,
                              'b (x w1) (y w2) e l -> b x y w1 w2 e l',
                              w1=self.window_size, w2=self.window_size)
        #[1，2，15，45，4，4，256]
        x = rearrange(x, 'b m d (x w1) (y w2) -> b m x y w1 w2 d',
                      w1=self.window_size, w2=self.window_size)
        #[1，2，15，45，4，4，256]
        x = self.window_attention(x, mask=mask_swap)
        x = self.window_ffd(x)
        #[1,2,256,60,180]
        x = rearrange(x, 'b m x y w1 w2 d -> b m d (x w1) (y w2)')


        mask_swap = mask
        # mask b h w 1 l -> b x y w1 w2 1 L
        #   [1,15,45,4,4,1,2]
        mask_swap = rearrange(mask_swap,
                              'b (x w1) (y w2) e l -> b x y w1 w2 e l',
                              w1=self.window_size, w2=self.window_size)
        #[1，2，15，45，4，4，256]
        x = rearrange(x, 'b m d (x w1) (y w2) -> b m x y w1 w2 d',
                      w1=self.window_size, w2=self.window_size)
        # x = rearrange(x, 'b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
        #               w1=self.window_size, w2=self.window_size)
        #[1，2，15，45，4，4，256]
        x = self.WindowAttentionGlobal(x, mask=mask_swap)

        x = self.windowGlobal_ffd(x)
        #[1,2,256,60,180]
        x = rearrange(x, 'b m x y w1 w2 d -> b m d (x w1) (y w2)')
        
        return x

class ERMVPFusionBlock(nn.Module):

    def __init__(self,
                 input_dim,
                 mlp_dim,
                 dim_head,
                 window_size,
                 agent_size,
                 drop_out):
        super(ERMVPFusionBlock, self).__init__()
        # b = batch * max_cav
        self.block = nn.Sequential(
            Rearrange('b m d (x w1) (y w2) -> b m x y w1 w2 d',
                      w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out,
                                                 agent_size, window_size)),
            PreNormResidual(input_dim,
                            FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange('b m x y w1 w2 d -> b m d (x w1) (y w2)'),

            Rearrange('b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                      w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out,
                                                 agent_size, window_size)),
            PreNormResidual(input_dim,
                            FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange('b m x y w1 w2 d -> b m d (w1 x) (w2 y)'),
        )

    def forward(self, x, mask=None):
        # todo: add mask operation later for mulit-agents
        x = self.block(x)
        return x

class ERMVPFusionEncoder(nn.Module):

    def __init__(self, args):
        super(ERMVPFusionEncoder, self).__init__()

        self.layers = nn.ModuleList([])
        self.depth = args['depth']

        # block related
        input_dim = args['input_dim']
        mlp_dim = args['mlp_dim']
        agent_size = args['agent_size']
        window_size = args['window_size']
        drop_out = args['drop_out']
        dim_head = args['dim_head']

        self.mask = False
        if 'mask' in args:
            self.mask = args['mask']

        for i in range(self.depth):
            if self.mask:
                block = ERMVPFusionBlockMask(input_dim,
                                    mlp_dim,
                                    dim_head,
                                    window_size,
                                    agent_size,
                                    drop_out)

            else:
                block = ERMVPFusionBlock(input_dim,
                                        mlp_dim,
                                        dim_head,
                                        window_size,
                                        agent_size,
                                        drop_out)
            self.layers.append(block)

        # mlp head
        self.mlp_head = nn.Sequential(
            Reduce('b m d h w -> b d h w', 'mean'),
            Rearrange('b d h w -> b h w d'),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange('b h w d -> b d h w')
        )

    def forward(self, x, mask=None):
        for stage in self.layers:
            x = stage(x, mask=mask)
        return self.mlp_head(x)

