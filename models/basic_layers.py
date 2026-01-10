import torch
from torch import nn, einsum
from einops import rearrange, repeat


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)
        return self.fn(q, k, v)

class PreNorm_hyper(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, h_dominate, h_a, h_v, h_hyper):
        h_dominate = self.norm1(h_dominate)
        h_a = self.norm2(h_a)
        h_v = self.norm3(h_v)
        h_hyper = self.norm4(h_hyper)
        return self.fn(h_dominate, h_a, h_v, h_hyper)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class HhyperLearningLayer(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_tv = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_tv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, h_dominate, h_a, h_v, h_hyper):
        h = self.heads

        q = self.to_q(h_dominate)
        k_a = self.to_k_ta(h_a)
        k_v = self.to_k_tv(h_v)

        v_a = self.to_v_ta(h_a)
        v_v = self.to_v_tv(h_v)

        q, k_a, k_v, v_a, v_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k_a, k_v, v_a, v_v))

        dots_q_ka = einsum('b h i d, b h j d -> b h i j', q, k_a) * self.scale
        attn_q_ka = self.attend(dots_q_ka)
        out_q_ka = einsum('b h i j, b h j d -> b h i d', attn_q_ka, v_a)
        out_q_ka = rearrange(out_q_ka, 'b h n d -> b n (h d)')

        dots_q_kv = einsum('b h i d, b h j d -> b h i j', q, k_v) * self.scale
        attn_q_kv = self.attend(dots_q_kv)
        out_q_kv = einsum('b h i j, b h j d -> b h i d', attn_q_kv, v_v)
        out_q_kv = rearrange(out_q_kv, 'b h n d -> b n (h d)')

        h_hyper_shift = self.to_out(out_q_ka + out_q_kv)
        h_hyper += h_hyper_shift

        return h_hyper


class HhyperLearningEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_hyper(dim, HhyperLearningLayer(dim, heads = heads, dim_head = dim_head, dropout = dropout))
            ]))

    def forward(self, h_domonate_list, h_a, h_v, h_hyper):
        for i, attn in enumerate(self.layers):
            h_hyper = attn[0](h_domonate_list[i], h_a, h_v, h_hyper)
        return h_hyper


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm_qkv(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, tgt, memory):
        for attn1, attn2, ff in self.layers:
            tgt = attn1(tgt, tgt, tgt) + tgt
            tgt = attn1(tgt, memory, memory) + tgt
            tgt = ff(tgt) + tgt
        return tgt


class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_qkv(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x


class Transformer(nn.Module):
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
             self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
             self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b = b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n+self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)

        return x


class CrossTransformer(nn.Module):
    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        
        # 新增：分层融合模块（Hierarchical Fusion）
class HierarchicalFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 底层：模态内局部特征编码（自注意力）
        self.intra_modal_encoder = nn.ModuleDict({
            'l': TransformerEncoder(
                dim=args['model']['dmml']['language_encoder']['input_dim'],
                depth=args['model']['dmml']['language_encoder']['depth'],
                heads=args['model']['dmml']['language_encoder']['heads'],
                dim_head=int(args['model']['dmml']['language_encoder']['input_dim']/args['model']['dmml']['language_encoder']['heads']),
                mlp_dim=args['model']['dmml']['language_encoder']['hidden_dim'],
                dropout=0.1
            ),
            'v': TransformerEncoder(
                dim=args['model']['dmml']['language_encoder']['input_dim'],
                depth=args['model']['dmml']['language_encoder']['depth'],
                heads=args['model']['dmml']['language_encoder']['heads'],
                dim_head=int(args['model']['dmml']['language_encoder']['input_dim']/args['model']['dmml']['language_encoder']['heads']),
                mlp_dim=args['model']['dmml']['language_encoder']['hidden_dim'],
                dropout=0.1
            ),
            'a': TransformerEncoder(
                dim=args['model']['dmml']['language_encoder']['input_dim'],
                depth=args['model']['dmml']['language_encoder']['depth'],
                heads=args['model']['dmml']['language_encoder']['heads'],
                dim_head=int(args['model']['dmml']['language_encoder']['input_dim']/args['model']['dmml']['language_encoder']['heads']),
                mlp_dim=args['model']['dmml']['language_encoder']['hidden_dim'],
                dropout=0.1
            )
        })
        
        # 中层：跨模态交互融合（交叉注意力）
        self.cross_modal_encoder = nn.ModuleList([
            CrossTransformerEncoder(
                dim=args['model']['dmml']['language_encoder']['input_dim'],
                depth=args['model']['dmml']['fuison_transformer']['depth'],
                heads=args['model']['dmml']['fuison_transformer']['heads'],
                dim_head=int(args['model']['dmml']['fuison_transformer']['input_dim']/args['model']['dmml']['fuison_transformer']['heads']),
                mlp_dim=args['model']['dmml']['fuison_transformer']['hidden_dim'],
                dropout=0.1
            ) for _ in range(2)  # 语言-视觉、语言-音频交叉
        ])
        
        # 高层：全局特征融合（池化 + 加权）
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fusion_linear = nn.Linear(
            args['model']['dmml']['language_encoder']['input_dim'] * 3,
            args['model']['dmml']['regression']['input_dim']
        )

    def forward(self, h_l, h_v, h_a, w):
        """
        h_l/h_v/h_a: 各模态底层特征 (b, n, d)
        w: 完整性校验分数 (b, 1)
        """
        # 1. 底层：模态内局部特征编码
        h_l_intra = self.intra_modal_encoder['l'](h_l)
        h_v_intra = self.intra_modal_encoder['v'](h_v)
        h_a_intra = self.intra_modal_encoder['a'](h_a)
        
        # 2. 中层：跨模态交互融合
        h_l_v = self.cross_modal_encoder[0](h_v_intra, h_l_intra)  # 语言为目标，视觉为源
        h_l_a = self.cross_modal_encoder[1](h_a_intra, h_l_intra)  # 语言为目标，音频为源
        h_cross = h_l_v + h_l_a + h_l_intra  # 融合跨模态特征
        
        # 3. 高层：全局特征融合 + 完整性加权
        # 全局池化
        h_l_global = self.global_pool(h_cross.transpose(1,2)).squeeze(-1)  # (b, d)
        h_v_global = self.global_pool(h_v_intra.transpose(1,2)).squeeze(-1)
        h_a_global = self.global_pool(h_a_intra.transpose(1,2)).squeeze(-1)
        
        # 基于完整性分数加权（缺失模态衰减）
        weight = w.expand(-1, h_l_global.shape[-1])  # (b, d)
        h_l_global = h_l_global * weight
        h_v_global = h_v_global * weight
        h_a_global = h_a_global * weight
        
        # 拼接 + 线性融合
        h_global = torch.cat([h_l_global, h_v_global, h_a_global], dim=-1)
        h_fused = self.fusion_linear(h_global)
        
        return h_fused
