import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 修正版 HU‑Guided Cross‑Attention（专为EAT优化）
# ============================================================
class HUDensityCrossAttention(nn.Module):
    """
    基于HU密度先验的交叉注意力模块
    专为心外膜脂肪组织（EAT）分割优化，精确匹配 HU ∈ [-190, -30] 窗
    """
    def __init__(
        self,
        dim=48,
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

    def compute_density_mask(self, ct_down):
        """
        计算EAT密度掩码，精确匹配 [-190, -30] HU
        输入: ct_down (B, 1, h, w, d)  归一化到[0,1]
        输出: mask (B, 1, h, w, d)     值域[0,1]
        """
        hu = ct_down * (self.hu_max - self.hu_min) + self.hu_min

        # 到窗中心的绝对距离
        dist_to_center = torch.abs(hu - self.window_center)

        # 内区间（窗内）掩码=1
        inner_mask = (dist_to_center <= self.half_width).float()

        # 外区间（超出过渡区）掩码=0
        outer_mask = (dist_to_center >= self.half_width + self.transition).float()

        # 过渡区掩码（仅用于平滑）
        transition_mask = 1.0 - inner_mask - outer_mask

        # 过渡区内归一化距离并计算平滑值
        d = dist_to_center - self.half_width  # [0, transition]
        norm_d = d / self.transition          # [0, 1]
        smooth_val = torch.sigmoid((1.0 - 2.0 * norm_d) * 5.0)

        mask = inner_mask + transition_mask * smooth_val
        # 注意：不添加额外的 unsqueeze，保持 (B, 1, h, w, d)
        return mask

    def forward(self, dec_feat, enc_feat, ct_down):
        B, C, h, w, d = dec_feat.shape

        # 1. 密度掩码 (B, 1, h, w, d)
        density_mask = self.compute_density_mask(ct_down)

        # 2. 编码器特征调制
        enc_feat_masked = enc_feat * (self.beta * density_mask + (1 - self.beta))

        # 3. 展平为序列
        # dec_feat: (B, C, h, w, d) -> (B, N, C)
        dec_flat = dec_feat.flatten(2).transpose(1, 2)
        enc_flat = enc_feat_masked.flatten(2).transpose(1, 2)
        mask_flat = density_mask.flatten(2).transpose(1, 2)  # (B, N, 1)

        # 形状断言（调试用）
        assert dec_flat.shape[-1] == C, f"dec_flat last dim {dec_flat.shape[-1]} != {C}"
        assert enc_flat.shape[-1] == C, f"enc_flat last dim {enc_flat.shape[-1]} != {C}"

        # 4. 线性投影
        Q = self.to_q(dec_flat)
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

        # 9. 恢复空间形状并残差连接
        out = out.transpose(1, 2).view(B, C, h, w, d)
        return dec_feat + out


# ============================================================
# 测试代码
# ============================================================
def test_eat_optimized_module():
    print("=" * 70)
    print("测试 EAT 优化版 HU‑Guided Cross‑Attention 模块")
    print("=" * 70)

    torch.manual_seed(42)

    batch_size = 2
    channels = 48
    spatial_size = (24, 24, 24)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    dec_feat = torch.randn(batch_size, channels, *spatial_size, device=device)
    enc_feat = torch.randn(batch_size, channels, *spatial_size, device=device)
    ct_down = torch.rand(batch_size, 1, *spatial_size, device=device)  # 归一化[0,1]

    model = HUDensityCrossAttention(
        dim=channels,
        num_heads=4,
        use_linear_attn=True
    ).to(device)

    # 测试1：前向传播形状
    print("\n--- 测试1：前向传播形状 ---")
    model.train()
    dec_feat.requires_grad_(True)
    enc_feat.requires_grad_(True)

    out = model(dec_feat, enc_feat, ct_down)
    print(f"输出形状: {out.shape}")
    assert out.shape == dec_feat.shape
    print("✓ 形状正确")

    # 测试2：梯度回传
    print("\n--- 测试2：梯度回传 ---")
    loss = out.mean()
    loss.backward()
    assert dec_feat.grad is not None and enc_feat.grad is not None
    print(f"解码器梯度范数: {dec_feat.grad.norm().item():.6f}")
    print(f"编码器梯度范数: {enc_feat.grad.norm().item():.6f}")
    print("✓ 梯度流正常")

    # 测试3：密度掩码精确性
    print("\n--- 测试3：密度掩码验证（EAT窗[-190, -30]） ---")
    test_hu = torch.tensor([-250, -200, -190, -150, -110, -50, -30, -20, 0, 40, 100]).float().to(device)
    test_ct = (test_hu - model.hu_min) / (model.hu_max - model.hu_min)
    test_ct = test_ct.view(1, 1, -1, 1, 1)

    with torch.no_grad():
        mask = model.compute_density_mask(test_ct)  # (1, 1, 11, 1, 1)
    mask_values = mask.flatten().cpu().numpy()

    print("HU值         掩码值       预期")
    for hu, m in zip(test_hu.cpu().numpy(), mask_values):
        if -190 <= hu <= -30:
            status = "✓ 窗内"
        elif hu < -200 or hu > 0:
            status = " 窗外"
        else:
            status = " 过渡"
        print(f"{hu:6.0f} HU  ->  {m:.4f}   {status}")

    # 严格断言
    assert mask_values[2] > 0.99, f"HU=-190应为1，实际{mask_values[2]:.4f}"
    assert mask_values[5] > 0.99, f"HU=-30应为1，实际{mask_values[5]:.4f}"
    assert mask_values[8] < 0.01, f"HU=0应为0，实际{mask_values[8]:.4f}"
    assert mask_values[9] < 0.01, f"HU=40（心肌）应为0，实际{mask_values[9]:.4f}"
    print("✓ 掩码精确匹配EAT窗，心肌区域响应为零")

    # 测试4：参数量
    print("\n--- 测试4：模块参数量 ---")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    print("✓ 轻量化设计")

    print("\n" + "=" * 70)
    print("所有测试通过！")
    print("=" * 70)


if __name__ == "__main__":
    test_eat_optimized_module()