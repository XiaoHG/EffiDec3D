import torch
import torch.nn as nn
import torch.nn.functional as F

class HUDensityCrossAttention(nn.Module):
    """
    基于HU密度先验的交叉注意力模块（假设解码器特征与编码器特征空间尺寸相同）
    """
    def __init__(
        self,
        dim,                    # 特征通道数
        num_heads=1,
        hu_min=-1000,
        hu_max=1000,
        window_lower=-190,
        window_upper=-30,
        transition=10.0,
        use_linear_attn=True
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 线性投影
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

        # HU参数
        self.register_buffer('hu_min', torch.tensor(hu_min))
        self.register_buffer('hu_max', torch.tensor(hu_max))

        # EAT窗参数
        self.window_lower = window_lower
        self.window_upper = window_upper
        self.window_center = (window_lower + window_upper) / 2.0
        self.half_width = (window_upper - window_lower) / 2.0
        self.transition = transition

        self.use_linear_attn = use_linear_attn

        # 可学习参数
        self.gamma = nn.Parameter(torch.tensor(10.0))
        self.tau = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.9))

    def compute_density_mask(self, ct):
        """ct: (B, 1, h, w, d) 归一化[0,1] -> mask: (B, 1, h, w, d)"""
        hu = ct * (self.hu_max - self.hu_min) + self.hu_min
        dist_to_center = torch.abs(hu - self.window_center)

        inner_mask = (dist_to_center <= self.half_width).float()
        outer_mask = (dist_to_center >= self.half_width + self.transition).float()
        transition_mask = 1.0 - inner_mask - outer_mask

        d = dist_to_center - self.half_width
        norm_d = d / self.transition
        smooth_val = torch.sigmoid((1.0 - 2.0 * norm_d) * 5.0)

        mask = inner_mask + transition_mask * smooth_val
        return mask  # (B, 1, h, w, d)

    def forward(self, dec_feat, enc_feat, ct_down):
        """
        dec_feat: (B, C, H, W, D)  上采样后的解码器特征
        enc_feat: (B, C, H, W, D)  编码器跳跃特征
        ct_down:  (B, 1, H, W, D)  与特征图同尺寸的CT图
        """
        B, C, H, W, D = dec_feat.shape

        # 1. 计算密度掩码
        density_mask = self.compute_density_mask(ct_down)  # (B, 1, H, W, D)

        # 2. 编码器特征调制
        enc_feat_masked = enc_feat * (self.beta * density_mask + (1 - self.beta))

        # 3. 展平为序列
        dec_flat = dec_feat.flatten(2).transpose(1, 2)  # (B, N, C)
        enc_flat = enc_feat_masked.flatten(2).transpose(1, 2)  # (B, N, C)
        mask_flat = density_mask.flatten(2).transpose(1, 2)  # (B, N, 1)

        # 4. 线性投影
        Q = self.to_q(dec_flat)  # (B, N, C)
        K = self.to_k(enc_flat)
        V = self.to_v(enc_flat)

        # 5. Query 调制
        q_mod_factor = torch.sigmoid(self.gamma * (mask_flat - self.tau))
        Q = Q * q_mod_factor

        # 6. 多头重塑
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, N, d_head)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 7. 注意力计算
        if self.use_linear_attn:
            Q_lin = F.elu(Q) + 1
            K_lin = F.elu(K) + 1
            KV = torch.einsum('b h n d, b h n e -> b h d e', K_lin, V)
            Z = K_lin.sum(dim=2, keepdim=True)
            out = torch.einsum('b h n d, b h d e -> b h n e', Q_lin, KV)
            denom = torch.einsum('b h n d, b h d -> b h n', Q_lin, Z.squeeze(2)).unsqueeze(-1)
            out = out / (denom + 1e-6)
        else:
            attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, V)

        # 8. 合并多头并投影
        out = out.transpose(1, 2).contiguous().view(B, -1, C)
        out = self.to_out(out)

        # 9. 恢复形状并残差连接
        out = out.transpose(1, 2).view(B, C, H, W, D)
        return dec_feat + out