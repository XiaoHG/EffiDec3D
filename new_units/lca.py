import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightCrossAttention(nn.Module):
    """
    轻量化交叉注意力模块 (LCA)
    用于U-Net跳跃连接，实现编码器特征(Fe)与解码器特征(Fd)的自适应融合。
    """
    def __init__(self, channels_encoder, channels_decoder, att_channels=None, reduction_ratio=2):
        super().__init__()
        
        if att_channels is None:
            att_channels = channels_decoder
        self.att_channels = att_channels
        
        # 编码器通道缩减（可选）
        if reduction_ratio > 1:
            self.encoder_reduce = nn.Sequential(
                nn.Conv3d(channels_encoder, channels_encoder // reduction_ratio, kernel_size=1),
                nn.BatchNorm3d(channels_encoder // reduction_ratio),
                nn.ReLU(inplace=True)
            )
            reduced_channels = channels_encoder // reduction_ratio
        else:
            self.encoder_reduce = nn.Identity()
            reduced_channels = channels_encoder
        
        # 投影层
        self.query_conv = nn.Conv3d(channels_decoder, att_channels, kernel_size=1)
        self.key_conv   = nn.Conv3d(reduced_channels, att_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(reduced_channels, att_channels, kernel_size=1)
        
        # 输出融合层
        self.out_conv = nn.Sequential(
            nn.Conv3d(att_channels + channels_decoder, channels_decoder, kernel_size=1),
            nn.BatchNorm3d(channels_decoder),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels_decoder, channels_decoder, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels_decoder),
            nn.ReLU(inplace=True)
        )
        
        self.scale = att_channels ** -0.5
        print(f"LCA: Encoder {channels_encoder} -> {reduced_channels} (after reduce), Decoder {channels_decoder} -> {att_channels}")
        
    def forward(self, fe, fd):
        batch_size, _, height, width, depth = fd.shape
        
        fe_reduced = self.encoder_reduce(fe)          # [B, reduced_c, H, W, D]
        
        Q = self.query_conv(fd)                       # [B, C_att, H, W, D]
        K = self.key_conv(fe_reduced)                 # [B, C_att, H, W, D]
        V = self.value_conv(fe_reduced)               # [B, C_att, H, W, D]
        
        # 展平空间维度
        Q = Q.view(batch_size, self.att_channels, -1)
        K = K.view(batch_size, self.att_channels, -1)
        V = V.view(batch_size, self.att_channels, -1)
        
        # 注意力计算
        attention_scores = torch.bmm(Q.transpose(1, 2), K) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        V_transposed = V.transpose(1, 2)
        attended_features = torch.bmm(attention_weights, V_transposed)
        
        # 恢复空间形状
        attended_features = attended_features.transpose(1, 2).contiguous()
        attended_features = attended_features.view(batch_size, self.att_channels, height, width, depth)
        
        # 融合
        combined = torch.cat([attended_features, fd], dim=1)
        output = self.out_conv(combined)
        return output


class DecoderBlockWithLCA(nn.Module):
    def __init__(self, in_channels_encoder, in_channels_decoder, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels_decoder, in_channels_decoder, kernel_size=2, stride=2)
        self.lca = LightweightCrossAttention(
            channels_encoder=in_channels_encoder,
            channels_decoder=in_channels_decoder,
            att_channels=in_channels_decoder // 2,
            reduction_ratio=2,
        )
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels_decoder, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, fe, fd):
        fd_up = self.up(fd)
        fd_fused = self.lca(fe, fd_up)
        out = self.conv_block(fd_fused)
        return out


if __name__ == "__main__":
    torch.manual_seed(42)
    B, C_enc, C_dec, H, W, D = 2, 256, 128, 32, 32, 32
    fe = torch.randn(B, C_enc, H, W, D)
    fd = torch.randn(B, C_dec, H//2, W//2, D//2)
    
    decoder_block = DecoderBlockWithLCA(
        in_channels_encoder=C_enc,
        in_channels_decoder=C_dec,
        out_channels=64
    )
    
    # 使用 thop 统计 FLOPs 和参数量
    try:
        from thop import profile, clever_format
        flops, params = profile(decoder_block, inputs=(fe, fd), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"\n=== FLOPs & Params ===")
        print(f"FLOPs : {flops}")
        print(f"Params: {params}")
    except ImportError:
        print("\n提示: 未安装 thop 库，无法自动统计 FLOPs。请运行 `pip install thop` 安装。")
        # 手动估算参数量
        params = sum(p.numel() for p in decoder_block.parameters())
        print(f"Params (manual): {params/1e6:.2f} M")
    
    output = decoder_block(fe, fd)
    print(f"\n输出特征形状: {output.shape}")