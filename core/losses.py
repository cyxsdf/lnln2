from torch import nn
from torch.nn import functional as F


class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 原有损失权重
        self.alpha = args['base']['alpha']
        self.beta = args['base']['beta']
        self.gamma = args['base']['gamma']
        self.sigma = args['base']['sigma']
        # 自监督损失权重
        self.mu = args['base']['mu']  # 掩码重建损失权重
        self.nu = args['base']['nu']  # 跨模态匹配损失权重
        
        self.CE_Fn = nn.CrossEntropyLoss()
        self.MSE_Fn = nn.MSELoss() 

    # 新增：模态内掩码重建损失
    def mask_reconstruction_loss(self, rec_feat, orig_feat, mask):
        """
        rec_feat: 重建的模态特征
        orig_feat: 原始未掩码的模态特征
        mask: 模态内掩码（1=掩码区域，0=未掩码区域）
        """
        # 仅计算掩码区域的重建损失
        loss_mask = self.MSE_Fn(rec_feat * mask, orig_feat * mask)
        return loss_mask
    
    # 新增：跨模态匹配损失
    def cross_modal_matching_loss(self, lang_feat, vis_feat, audio_feat, match_label):
        """
        lang_feat/vis_feat/audio_feat: 各模态全局特征
        match_label: 匹配标签（1=对应，0=不对应）
        """
        # 计算语言-视觉、语言-音频的相似度
        sim_vis = torch.cosine_similarity(lang_feat, vis_feat, dim=-1)
        sim_audio = torch.cosine_similarity(lang_feat, audio_feat, dim=-1)
        sim_concat = torch.stack([sim_vis, sim_audio], dim=-1)  # (b, 2)
        # 交叉熵损失（分类是否匹配）
        loss_match = self.CE_Fn(sim_concat, match_label)
        return loss_match

    def forward(self, out, label, is_pretrain=False):
        if is_pretrain:
            # 预训练阶段：仅优化自监督损失
            l_mask = self.mask_reconstruction_loss(out['rec_mask_feat'], out['orig_feat'], out['mask'])
            l_match = self.cross_modal_matching_loss(out['lang_global'], out['vis_global'], out['audio_global'], label['match_label'])
            total_loss = self.mu * l_mask + self.nu * l_match
            return {'loss': total_loss, 'l_mask': l_mask, 'l_match': l_match}
        else:
            # 微调阶段：原有损失 + 少量自监督损失（可选）
            l_cc = self.MSE_Fn(out['w'], label['completeness_labels']) if out['w'] is not None else 0
            l_adv = self.CE_Fn(out['effectiveness_discriminator_out'], label['effectiveness_labels']) if out['effectiveness_discriminator_out'] is not None else 0
            l_rec = self.MSE_Fn(out['rec_feats'], out['complete_feats']) if out['rec_feats'] is not None and out['complete_feats'] is not None else 0
            l_sp = self.MSE_Fn(out['sentiment_preds'], label['sentiment_labels'])
            
            total_loss = self.alpha * l_cc + self.beta * l_adv + self.gamma * l_rec + self.sigma * l_sp
            return {'loss': total_loss, 'l_sp': l_sp, 'l_cc': l_cc, 'l_adv': l_adv, 'l_rec': l_rec}
