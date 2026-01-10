import torch
from torch import nn
from .basic_layers import Transformer, CrossTransformer, GradientReversalLayer, HierarchicalFusion
from .bert import BertTextEncoder
from einops import rearrange, repeat


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

        # 替换原有DMML为分层融合模块
        self.hierarchical_fusion = HierarchicalFusion(args)
        self.regression_head = nn.Linear(args['model']['dmml']['regression']['input_dim'], args['model']['dmml']['regression']['out_dim'])

        # 自监督预训练相关：掩码生成器
        self.mask_generator = nn.Parameter(torch.randn(1, args['model']['feature_extractor']['token_length'][0], 128))  # 模态内掩码
        self.match_projection = nn.Linear(128, 64)  # 跨模态匹配投影

    def forward(self, complete_input, incomplete_input, is_pretrain=False):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        b = vision_m.size(0)

        h_1_v = self.proj_v(vision_m)[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]
        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]

        feat_tmp = self.completeness_check[0](h_1_l)[:, :1].squeeze()
        w = self.completeness_check[1](feat_tmp) # 完整性分数

        h_0_p = repeat(self.h_p, '1 n d -> b n d', b = b)
        h_1_p = self.proxy_dominate_modality_generator(torch.cat([h_0_p, h_1_a, h_1_v], dim=1))[:, :8]
        h_1_p = self.GRL(h_1_p)
        h_1_d = h_1_p * (1-w.unsqueeze(-1)) + h_1_l * w.unsqueeze(-1)

        # 自监督预训练分支
        if is_pretrain:
            # 任务1：模态内掩码预测
            mask = (torch.rand_like(h_1_l) > 0.1).float()  # 10%掩码率
            h_1_l_masked = h_1_l * (1 - mask) + self.mask_generator * mask
            rec_mask_feat = self.reconstructor[2](h_1_l_masked)  # 重建掩码特征
            
            # 任务2：跨模态匹配
            lang_proj = self.match_projection(torch.mean(h_1_l, dim=1))
            vis_proj = self.match_projection(torch.mean(h_1_v, dim=1))
            audio_proj = self.match_projection(torch.mean(h_1_a, dim=1))
            
            # 返回自监督所需特征
            return {
                'rec_mask_feat': rec_mask_feat,
                'orig_feat': h_1_l,
                'mask': mask,
                'lang_global': lang_proj,
                'vis_global': vis_proj,
                'audio_global': audio_proj,
                'w': w
            }

        # 正常训练/推理：分层融合
        h_fused = self.hierarchical_fusion(h_1_d, h_1_v, h_1_a, w)
        output = self.regression_head(h_fused)

        rec_feats, complete_feats, effectiveness_discriminator_out = None, None, None
        if (vision is not None) and (audio is not None) and (language is not None):
            # 重建分支
            rec_feat_a = self.reconstructor[0](h_1_a)
            rec_feat_v = self.reconstructor[1](h_1_v)
            rec_feat_l = self.reconstructor[2](h_1_l)
            rec_feats = torch.cat([rec_feat_a, rec_feat_v, rec_feat_l], dim=1)

            # 完整特征
            complete_language_feat = self.proj_l(self.bertmodel(language))[:, :8]
            complete_vision_feat = self.proj_v(vision)[:, :8]
            complete_audio_feat = self.proj_a(audio)[:, :8]

            effective_discriminator_input = rearrange(torch.cat([h_1_d, complete_language_feat]), 'b n d -> (b n) d')
            effectiveness_discriminator_out = self.effective_discriminator(effective_discriminator_input)
        
            complete_feats = torch.cat([complete_audio_feat, complete_vision_feat, complete_language_feat], dim=1)

        return {
            'sentiment_preds': output, 
            'w': w, 
            'effectiveness_discriminator_out': effectiveness_discriminator_out, 
            'rec_feats': rec_feats, 
            'complete_feats': complete_feats
        }

def build_model(args):
    return LNLN(args)
