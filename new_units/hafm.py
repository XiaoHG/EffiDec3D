import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile  # 用于计算FLOPs和参数量

class HUAwareModulation(nn.Module):
    """
    HU感知特征调制模块 (HAFM)
    输入: 原始CT图像 (x_ct) 和 对应的编码器特征图 (x_feat)
    输出: 经过HU先验调制的特征图
    原理: 生成一个与EAT HU范围相关的注意力图，用于加权特征图。
    """
    def __init__(self, in_channels, hu_low=-190, hu_high=-30, temperature=0.1):
        super().__init__()
        self.hu_low = hu_low
        self.hu_high = hu_high
        self.temperature = temperature
        
        # 一个小型网络，学习如何根据HU信息生成调制权重
        self.mapper = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 初始化参数
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def generate_hu_prior_mask(self, x_ct):
        """
        生成基于HU阈值的先验掩码
        假设x_ct已经是原始的HU值（未归一化）
        """
        # 创建EAT概率图：在[-190, -30]区间内权重高，区间外权重低
        # 使用高斯函数平滑过渡
        hu_center = (self.hu_low + self.hu_high) / 2  # -110
        hu_width = (self.hu_high - self.hu_low) / 2   # 80
        
        # 计算每个体素属于EAT的概率（基于HU值）
        # 使用高斯核：exp(-(hu - center)^2 / (2 * width^2))
        hu_diff = (x_ct - hu_center) / hu_width
        prior_weight = torch.exp(-hu_diff ** 2 / 2)
        
        # 确保权重在[0.2, 0.8]范围内
        prior_weight = 0.2 + 0.6 * prior_weight
        
        return prior_weight
    
    def forward(self, x_feat, x_ct):
        """
        Args:
            x_feat: 编码器特征 [B, C, H, W, D]
            x_ct: 原始CT图像（假设为原始HU值）[B, 1, H, W, D]
        Returns:
            调制后的特征 [B, C, H, W, D]
        """
        B, C, H, W, D = x_feat.shape
        
        # 1. 对CT图像进行下采样以匹配特征图尺寸
        if x_ct.shape[2:] != (H, W, D):
            x_ct_down = F.interpolate(x_ct, size=(H, W, D), mode='trilinear', align_corners=False)
        else:
            x_ct_down = x_ct
        
        # 2. 生成基于学习的调制权重
        learned_weight = self.mapper(x_ct_down)  # [B, C, H, W, D]
        
        # 3. 生成基于HU先验的固定掩码
        prior_weight = self.generate_hu_prior_mask(x_ct_down)  # [B, 1, H, W, D]
        prior_weight = prior_weight.expand(-1, C, -1, -1, -1)  # [B, C, H, W, D]
        
        # 4. 融合学习权重和先验权重
        # 训练初期更多依赖先验，后期更多依赖学习
        if self.training:
            # 使用温度参数控制融合程度
            alpha = torch.sigmoid(torch.tensor(self.temperature))
            final_weight = alpha * learned_weight + (1 - alpha) * prior_weight
        else:
            # 推理时主要使用学习到的权重
            final_weight = learned_weight
        
        # 5. 应用调制：特征图与权重图逐元素相乘
        modulated_feat = x_feat * final_weight
        
        # 6. 残差连接，保留原始特征信息流
        output = modulated_feat + x_feat
        
        return output


def calculate_params_flops(model, input_size_feat, input_size_ct):
    """
    计算模型的参数量和FLOPs
    """
    # 创建模拟输入
    dummy_feat = torch.randn(*input_size_feat)
    dummy_ct = torch.randn(*input_size_ct)
    
    # 计算FLOPs和参数量
    flops, params = profile(model, inputs=(dummy_feat, dummy_ct), verbose=False)
    
    return flops, params


def test_hafm_module():
    """测试HAFM模块的功能和性能"""
    print("=" * 60)
    print("HU感知特征调制模块 (HAFM) 测试")
    print("=" * 60)
    
    # 测试配置
    batch_size = 2
    in_channels = 48
    spatial_size = (48, 48, 48)  # H, W, D
    
    # 创建模拟输入
    print(f"\n1. 创建模拟输入:")
    print(f"   Batch size: {batch_size}")
    print(f"   输入通道数: {in_channels}")
    print(f"   空间尺寸: {spatial_size}")
    
    # 模拟编码器特征（随机值）
    x_feat = torch.randn(batch_size, in_channels, *spatial_size)
    print(f"   编码器特征形状: {x_feat.shape}")
    
    # 模拟CT图像（HU值范围大致在[-200, 50]之间）
    # 这里我们模拟一些在EAT范围内的HU值
    hu_min, hu_max = -200, 50
    x_ct = torch.rand(batch_size, 1, *spatial_size) * (hu_max - hu_min) + hu_min
    print(f"   CT图像形状: {x_ct.shape}")
    print(f"   CT值范围: [{x_ct.min():.1f}, {x_ct.max():.1f}] HU")
    
    # 创建HAFM模块
    print(f"\n2. 创建HAFM模块:")
    hafm = HUAwareModulation(in_channels=in_channels, hu_low=-190, hu_high=-30)
    print(f"   HU阈值范围: [{hafm.hu_low}, {hafm.hu_high}] HU")
    print(f"   温度参数: {hafm.temperature}")
    
    # 前向传播测试
    print(f"\n3. 前向传播测试:")
    with torch.no_grad():
        output = hafm(x_feat, x_ct)
    print(f"   输入特征形状: {x_feat.shape}")
    print(f"   输出特征形状: {output.shape}")
    print(f"   输入输出形状一致: {x_feat.shape == output.shape}")
    
    # 检查输出值范围
    print(f"\n4. 输出值检查:")
    print(f"   输入特征均值: {x_feat.mean():.4f}, 标准差: {x_feat.std():.4f}")
    print(f"   输出特征均值: {output.mean():.4f}, 标准差: {output.std():.4f}")
    
    # 计算参数量和FLOPs
    print(f"\n5. 计算复杂度:")
    
    # 方法1: 使用thop库（需要安装: pip install thop）
    try:
        input_size_feat = (batch_size, in_channels, *spatial_size)
        input_size_ct = (batch_size, 1, *spatial_size)
        flops, params = calculate_params_flops(hafm, input_size_feat, input_size_ct)
        
        print(f"   总参数量: {params:,} ({params/1e6:.2f} M)")
        print(f"   总FLOPs: {flops:,} ({flops/1e9:.2f} G)")
        
        # 分解各个部分的参数量
        print(f"\n   参数量分解:")
        total_params = 0
        for name, module in hafm.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                module_params = sum(p.numel() for p in module.parameters())
                if module_params > 0:
                    print(f"     {name}: {module_params:,} ({module_params/1e6:.3f} M)")
                    total_params += module_params
        
        # 计算相对于输入特征图的参数量占比
        feat_params_approx = in_channels * 3 * 3 * 3 * in_channels * 2  # 两个3x3x3卷积的近似
        ratio = params / feat_params_approx * 100
        print(f"\n   HAFM参数量占典型卷积块的: {ratio:.1f}%")
        
    except ImportError:
        print("   警告: 未安装thop库，无法计算FLOPs")
        print("   请安装: pip install thop")
        
        # 手动计算参数量
        params = sum(p.numel() for p in hafm.parameters())
        print(f"   总参数量: {params:,} ({params/1e6:.2f} M)")
    
    # 测试先验掩码生成
    print(f"\n6. 测试HU先验掩码生成:")
    with torch.no_grad():
        prior_mask = hafm.generate_hu_prior_mask(x_ct)
        print(f"   先验掩码形状: {prior_mask.shape}")
        print(f"   先验掩码值范围: [{prior_mask.min():.3f}, {prior_mask.max():.3f}]")
        
        # 统计不同HU值范围的权重
        hu_ranges = [(-200, -190), (-190, -30), (-30, 50)]
        for hu_min, hu_max in hu_ranges:
            mask = (x_ct >= hu_min) & (x_ct <= hu_max)
            if mask.sum() > 0:
                avg_weight = prior_mask[mask].mean()
                print(f"   HU范围[{hu_min}, {hu_max}]: 平均权重 = {avg_weight:.3f}")
    
    # 性能测试（推理时间）
    print(f"\n7. 性能测试（推理时间）:")
    import time
    
    # 预热
    for _ in range(10):
        _ = hafm(x_feat, x_ct)
    
    # 正式测试
    num_iterations = 100
    start_time = time.time()
    for _ in range(num_iterations):
        _ = hafm(x_feat, x_ct)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations * 1000  # ms
    fps = 1000 / avg_time if avg_time > 0 else float('inf')
    
    print(f"   平均推理时间: {avg_time:.2f} ms")
    print(f"   推理速度: {fps:.1f} FPS")
    
    # 内存占用测试
    print(f"\n8. 内存占用测试:")
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if torch.cuda.is_available():
        x_feat_cuda = x_feat.cuda()
        x_ct_cuda = x_ct.cuda()
        hafm_cuda = hafm.cuda()
        
        # 记录初始内存
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        
        # 前向传播
        output_cuda = hafm_cuda(x_feat_cuda, x_ct_cuda)
        
        # 记录峰值内存
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"   GPU初始内存: {initial_memory:.2f} MB")
        print(f"   GPU峰值内存: {peak_memory:.2f} MB")
        print(f"   HAFM模块内存占用: {peak_memory - initial_memory:.2f} MB")
    else:
        print("   GPU不可用，跳过内存测试")
    
    print(f"\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return hafm, x_feat, x_ct, output


if __name__ == "__main__":
    # 运行测试
    hafm_module, feat_input, ct_input, feat_output = test_hafm_module()
    
    # 额外的集成测试示例
    print(f"\n\n附加测试: 在编码器块中集成HAFM")
    print("-" * 40)
    
    class SimpleEncoderBlockWithHAFM(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm3d(out_channels)
            
            # 插入HAFM模块
            self.hafm = HUAwareModulation(in_channels=out_channels)
            
        def forward(self, x, ct_image):
            identity = x if x.shape[1] == self.conv1.out_channels else \
                      nn.Conv3d(x.shape[1], self.conv1.out_channels, kernel_size=1)(x)
            
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            
            # 应用HU感知调制
            out = self.hafm(out, ct_image)
            
            out += identity
            out = self.relu(out)
            return out
    
    # 测试集成块
    encoder_block = SimpleEncoderBlockWithHAFM(in_channels=64, out_channels=128)
    print(f"编码器块参数量: {sum(p.numel() for p in encoder_block.parameters()):,}")
    
    # 计算集成块的FLOPs（如果thop可用）
    try:
        from thop import profile
        dummy_input = torch.randn(2, 64, 32, 32, 32)
        dummy_ct = torch.randn(2, 1, 32, 32, 32)
        flops, params = profile(encoder_block, inputs=(dummy_input, dummy_ct), verbose=False)
        print(f"编码器块FLOPs: {flops/1e9:.2f} G")
        
        # 计算HAFM模块在整体中的占比
        hafm_flops, hafm_params = calculate_params_flops(
            encoder_block.hafm, 
            (2, 128, 32, 32, 32),
            (2, 1, 32, 32, 32)
        )
        print(f"HAFM占编码器块参数量: {hafm_params/params*100:.1f}%")
        print(f"HAFM占编码器块FLOPs: {hafm_flops/flops*100:.1f}%")
        
    except ImportError:
        print("thop库未安装，跳过FLOPs计算")
