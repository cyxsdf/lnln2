import torch
from torch import nn
from .basic_layers import Transformer, HhyperLearningEncoder, GradientReversalLayer
from .bert import BertTextEncoder
from einops import rearrange, repeat


class CausalHierarchicalFusion(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        # 1. 因果注意力层（底层）
        self.causal_attention = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
        # 2. 局部时序对齐层（中层）
        self.local_align = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1) for _ in range(3)  # 文本/视觉/音频
        ])
        # 3. 全局融合层（顶层）
        self.global_fusion = nn.Sequential(
            nn.Linear(d_model*3, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, l_feat, v_feat, a_feat):
        # 1. 局部时序对齐：卷积捕捉局部依赖
        l_feat_local = self.local_align[0](l_feat.permute(0,2,1)).permute(0,2,1)
        v_feat_local = self.local_align[1](v_feat.permute(0,2,1)).permute(0,2,1)
        a_feat_local = self.local_align[2](a_feat.permute(0,2,1)).permute(0,2,1)
        
        # 2. 因果注意力：仅允许文本→视觉/音频的因果依赖
        # 构建因果掩码（文本为query，视觉/音频为key/value）
        l_len, v_len = l_feat_local.shape[1], v_feat_local.shape[1]
        a_len = a_feat_local.shape[1]
        causal_mask_v = torch.triu(torch.ones(l_len, v_len), diagonal=1).bool().to(l_feat.device)
        causal_mask_a = torch.triu(torch.ones(l_len, a_len), diagonal=1).bool().to(l_feat.device)
        
        # 调整维度以适应MultiheadAttention (seq_len, batch, dim)
        l_q = l_feat_local.permute(1, 0, 2)
        v_kv = v_feat_local.permute(1, 0, 2)
        a_kv = a_feat_local.permute(1, 0, 2)
        
        # 文本引导视觉融合
        v_fused, _ = self.causal_attention(
            query=l_q, key=v_kv, value=v_kv,
            attn_mask=causal_mask_v
        )
        v_fused = v_fused.permute(1, 0, 2)  # 恢复 (batch, seq_len, dim)
        
        # 文本引导音频融合
        a_fused, _ = self.causal_attention(
            query=l_q, key=a_kv, value=a_kv,
            attn_mask=causal_mask_a
        )
        a_fused = a_fused.permute(1, 0, 2)  # 恢复 (batch, seq_len, dim)
        
        # 3. 全局融合
        l_global = l_feat_local.mean(dim=1)  # (b, d)
        v_global = v_fused.mean(dim=1)
        a_global = a_fused.mean(dim=1)
        fused_feat = self.global_fusion(torch.cat([l_global, v_global, a_global], dim=-1))
        
        return fused_feat


class LNLN(nn.Module):
    def __init__(self, args):
        super(LNLN, self).__init__()

        self.h_hyper = nn.Parameter(torch.ones(1, args['model']['feature_extractor']['token_length'][0], 128))
        self.h_p = nn.Parameter(torch.ones(1, args['model']['feature_extractor']['token_length'][0], 128))

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args['model']['feature_extractor']['bert_pretrained'])

        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0], args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][0], 
                        dim=args['model']['feature_extractor']['hidden_dims'][0], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][0])
        )

        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2], args['model']['feature_extractor']['hidden_dims'][2]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][2], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][2], 
                        dim=args['model']['feature_extractor']['hidden_dims'][2], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][2])
        )

        self.proj_v = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][1], args['model']['feature_extractor']['hidden_dims'][1]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][1], 
                        dim=args['model']['feature_extractor']['hidden_dims'][1], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][1])
        )
        
        
        self.proxy_dominate_modality_generator = Transformer(
            num_frames=args['model']['dmc']['proxy_dominant_feature_generator']['input_length'], 
            save_hidden=False, 
            token_len=args['model']['dmc']['proxy_dominant_feature_generator']['token_length'], 
            dim=args['model']['dmc']['proxy_dominant_feature_generator']['input_dim'], 
            depth=args['model']['dmc']['proxy_dominant_feature_generator']['depth'], 
            heads=args['model']['dmc']['proxy_dominant_feature_generator']['heads'], 
            mlp_dim=args['model']['dmc']['proxy_dominant_feature_generator']['hidden_dim'])
        
        self.GRL = GradientReversalLayer(alpha=1.0)

        self.effective_discriminator = nn.Sequential(
            nn.Linear(args['model']['dmc']['effectiveness_discriminator']['input_dim'], 
                      args['model']['dmc']['effectiveness_discriminator']['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.Linear(args['model']['dmc']['effectiveness_discriminator']['hidden_dim'], 
                      args['model']['dmc']['effectiveness_discriminator']['out_dim']),
        )

        self.completeness_check = nn.ModuleList([
            Transformer(num_frames=args['model']['dmc']['completeness_check']['input_length'], 
                        save_hidden=False, 
                        token_len=args['model']['dmc']['completeness_check']['token_length'], 
                        dim=args['model']['dmc']['completeness_check']['input_dim'], 
                        depth=args['model']['dmc']['completeness_check']['depth'], 
                        heads=args['model']['dmc']['completeness_check']['heads'], 
                        mlp_dim=args['model']['dmc']['completeness_check']['hidden_dim']),

            nn.Sequential(
                nn.Linear(args['model']['dmc']['completeness_check']['hidden_dim'], int(args['model']['dmc']['completeness_check']['hidden_dim']/2)),
                nn.LeakyReLU(0.1),
                nn.Linear(int(args['model']['dmc']['completeness_check']['hidden_dim']/2), 1),
                nn.Sigmoid()),
        ])


        self.reconstructor = nn.ModuleList([
            Transformer(num_frames=args['model']['reconstructor']['input_length'], 
                        save_hidden=False, 
                        token_len=None, 
                        dim=args['model']['reconstructor']['input_dim'], 
                        depth=args['model']['reconstructor']['depth'], 
                        heads=args['model']['reconstructor']['heads'], 
                        mlp_dim=args['model']['reconstructor']['hidden_dim']) for _ in range(3)
        ])


        self.dmml = nn.ModuleList([
            Transformer(num_frames=args['model']['dmml']['language_encoder']['input_length'], 
                        save_hidden=True, 
                        token_len=None, 
                        dim=args['model']['dmml']['language_encoder']['input_dim'], 
                        depth=args['model']['dmml']['language_encoder']['depth'], 
                        heads=args['model']['dmml']['language_encoder']['heads'], 
                        mlp_dim=args['model']['dmml']['language_encoder']['hidden_dim']),

            HhyperLearningEncoder(dim=args['model']['dmml']['hyper_modality_learning']['input_dim'], 
                                  dim_head=int(args['model']['dmml']['hyper_modality_learning']['input_dim']/args['model']['dmml']['hyper_modality_learning']['heads']),
                                  depth=args['model']['dmml']['hyper_modality_learning']['depth'], 
                                  heads=args['model']['dmml']['hyper_modality_learning']['heads']),

            # 替换为因果层次化融合器
            CausalHierarchicalFusion(
                d_model=args['model']['dmml']['fuison_transformer']['input_dim'],
                nhead=args['model']['dmml']['fuison_transformer']['heads']
            ),

            nn.Linear(args['model']['dmml']['regression']['input_dim'], args['model']['dmml']['regression']['out_dim'])
        ])



    def forward(self, complete_input, incomplete_input):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        b = vision_m.size(0)

        h_1_v = self.proj_v(vision_m)[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]

        feat_tmp = self.completeness_check[0](h_1_l)[:, :1].squeeze()
        w = self.completeness_check[1](feat_tmp) # completeness scores

        h_0_p = repeat(self.h_p, '1 n d -> b n d', b = b)
        h_1_p = self.proxy_dominate_modality_generator(torch.cat([h_0_p, h_1_a, h_1_v], dim=1))[:, :8]
        h_1_p = self.GRL(h_1_p)
        h_1_d = h_1_p * (1-w.unsqueeze(-1)) + h_1_l * w.unsqueeze(-1)

        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b = b)
        h_d_list = self.dmml[0](h_1_d)
        h_hyper = self.dmml[1](h_d_list, h_1_a, h_1_v, h_hyper)
        
        # 使用因果层次化融合器进行融合
        feat = self.dmml[2](h_1_l, h_1_v, h_1_a)  # 文本/视觉/音频特征
        
        # 使用融合特征进行情感预测
        output = self.dmml[3](feat)

        rec_feats, complete_feats, effectiveness_discriminator_out = None, None, None
        if (vision is not None) and (audio is not None) and (language is not None):
            # Reconstruction
            rec_feat_a = self.reconstructor[0](h_1_a)
            rec_feat_v = self.reconstructor[1](h_1_v)
            rec_feat_l = self.reconstructor[2](h_1_l)
            rec_feats = torch.cat([rec_feat_a, rec_feat_v, rec_feat_l], dim=1)

            # Compute the complete features as the label of reconstruction
            complete_language_feat = self.proj_l(self.bertmodel(language))[:, :8]
            complete_vision_feat = self.proj_v(vision)[:, :8]
            complete_audio_feat = self.proj_a(audio)[:, :8]

            effective_discriminator_input = rearrange(torch.cat([h_1_d, complete_language_feat]), 'b n d -> (b n) d')
            effectiveness_discriminator_out = self.effective_discriminator(effective_discriminator_input)
        
            complete_feats = torch.cat([complete_audio_feat, complete_vision_feat, complete_language_feat], dim=1) # as the label of reconstruction
  

        return {'sentiment_preds': output, 
                'w': w, 
                'effectiveness_discriminator_out': effectiveness_discriminator_out, 
                'rec_feats': rec_feats, 
                'complete_feats': complete_feats}



def build_model(args):
    return LNLN(args)
