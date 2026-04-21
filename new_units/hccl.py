import torch
import torch.nn as nn
import torch.nn.functional as F

class HUConsistencyLoss(nn.Module):
    """
    HU一致性约束损失 (HCCL)
    鼓励模型预测的EAT区域的平均CT值落在先验合理范围内[-190, -30] HU。
    
    设计原理:
    1. EAT在CT上有明确的HU范围[-190, -30]
    2. 预测的EAT区域平均HU应接近该范围的中心值(-110)
    3. 通过约束HU一致性，使分割结果在物理属性上也正确
    """
    def __init__(self, 
                 target_hu_mean=-110.0, 
                 hu_tolerance=40.0, 
                 loss_weight=0.1,
                 hu_range=(-190, -30)):
        """
        Args:
            target_hu_mean: 期望的EAT平均HU值（-190和-30的中点）
            hu_tolerance: 允许偏离目标值的容忍范围（HU单位）
            loss_weight: 该损失项的权重
            hu_range: EAT的HU有效范围
        """
        super().__init__()
        self.target_mean = target_hu_mean
        self.tolerance = hu_tolerance
        self.weight = loss_weight
        self.hu_low, self.hu_high = hu_range
        
    def forward(self, pred_prob, ct_image):
        """
        Args:
            pred_prob: 模型预测的概率图 [B, 1, D, H, W]，值在[0,1]之间
            ct_image: 对应的原始CT图像（必须是原始HU值）[B, 1, D, H, W]
        Returns:
            loss: HU一致性损失值
        """
        batch_size = pred_prob.shape[0]  # 获取批次大小
        
        # 1. 将概率图转换为二值掩码
        binary_mask = (pred_prob > 0.5).float()
        
        # 2. 计算每个样本的HU统计和损失
        total_loss = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            # 提取当前样本的掩码和CT
            mask_sample = binary_mask[i:i+1]  # [1, 1, D, H, W]
            ct_sample = ct_image[i:i+1]       # [1, 1, D, H, W]
            
            # 计算预测EAT区域内的HU值
            masked_ct = ct_sample * mask_sample
            sum_hu = torch.sum(masked_ct)
            num_voxels = torch.sum(mask_sample)
            
            eps = 1e-6
            if num_voxels > eps:
                # 计算平均HU值
                mean_hu = sum_hu / num_voxels
                
                # 计算在有效HU范围内的体素比例
                in_range_mask = (ct_sample >= self.hu_low) & (ct_sample <= self.hu_high)
                valid_voxels = torch.sum(mask_sample * in_range_mask.float())
                validity_ratio = valid_voxels / num_voxels
                
                # 计算损失（三项组合）
                # 1) Huber损失：鼓励平均HU接近目标值
                # 修复：确保mean_hu和target_tensor形状一致
                target_tensor = torch.tensor([self.target_mean], device=pred_prob.device, dtype=mean_hu.dtype)
                # 扩展mean_hu以匹配target_tensor的形状
                mean_hu_expanded = mean_hu.view(1)  # 将标量转换为形状
                huber_loss = F.smooth_l1_loss(mean_hu_expanded, target_tensor)
                
                # 2) 容忍损失：超过容忍范围时惩罚
                diff = torch.abs(mean_hu - self.target_mean)
                tolerance_loss = torch.max(diff - self.tolerance, torch.zeros_like(diff))
                
                # 3) 有效性惩罚：鼓励更多体素在有效范围内
                validity_penalty = 1.0 - validity_ratio
                
                # 综合损失
                sample_loss = huber_loss + 0.3 * tolerance_loss + 0.2 * validity_penalty
                total_loss += sample_loss
                valid_samples += 1
                
                # 调试信息
                if self.training:  # 只在训练时打印，避免测试时干扰
                    print(f"[HCCL] 样本{i}: "
                          f"mean_HU={mean_hu.item():.1f}, "
                          f"体素数={num_voxels.item():.0f}, "
                          f"有效比例={validity_ratio.item():.3f}, "
                          f"损失={sample_loss.item():.4f}")
            else:
                # 如果没有预测到EAT，给予一个固定惩罚
                no_prediction_penalty = torch.tensor([0.5], device=pred_prob.device)
                total_loss += no_prediction_penalty
        
        # 3. 计算平均损失并应用权重
        if valid_samples > 0:
            avg_loss = total_loss / valid_samples
        else:
            avg_loss = total_loss / batch_size
            
        weighted_loss = self.weight * avg_loss
        
        return weighted_loss


def test_hccl_module():
    """测试HCCL模块的功能"""
    print("=" * 60)
    print("HU一致性约束损失 (HCCL) 模块测试")
    print("=" * 60)
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 测试配置
    batch_size = 3
    spatial_size = (32, 32, 16)  # D, H, W
    
    print(f"\n1. 测试配置:")
    print(f"   批大小: {batch_size}")
    print(f"   空间尺寸: {spatial_size}")
    
    # 创建模拟数据
    print(f"\n2. 创建模拟数据:")
    
    # 模拟CT图像（原始HU值）
    ct_images = []
    pred_probs = []
    
    # 样本0: "完美"的EAT分割 - HU值在有效范围内
    ct0 = torch.randn(1, 1, *spatial_size) * 30 - 110  # 均值-110，标准差30
    prob0 = torch.zeros(1, 1, *spatial_size)
    prob0[:, :, 10:22, 10:22, 5:11] = 0.9  # 模拟EAT区域
    ct_images.append(ct0)
    pred_probs.append(prob0)
    
    # 样本1: "一般"的EAT分割 - 部分超出范围
    ct1 = torch.randn(1, 1, *spatial_size) * 50 - 90   # 均值-90，标准差50
    prob1 = torch.zeros(1, 1, *spatial_size)
    prob1[:, :, 12:24, 12:24, 6:12] = 0.7
    ct_images.append(ct1)
    pred_probs.append(prob1)
    
    # 样本2: "错误"的EAT分割 - HU值偏高
    ct2 = torch.randn(1, 1, *spatial_size) * 30 - 50   # 均值-50，标准差30
    prob2 = torch.zeros(1, 1, *spatial_size)
    prob2[:, :, 14:26, 14:26, 7:13] = 0.8
    ct_images.append(ct2)
    pred_probs.append(prob2)
    
    # 合并批次
    ct_batch = torch.cat(ct_images, dim=0)
    prob_batch = torch.cat(pred_probs, dim=0)
    
    print(f"   CT图像形状: {ct_batch.shape}")
    print(f"   预测概率形状: {prob_batch.shape}")
    print(f"   CT值范围: [{ct_batch.min():.1f}, {ct_batch.max():.1f}] HU")
    
    # 创建HCCL模块
    print(f"\n3. 创建HCCL模块:")
    hccl = HUConsistencyLoss(
        target_hu_mean=-110.0,
        hu_tolerance=40.0,
        loss_weight=0.1,
        hu_range=(-190, -30)
    )
    
    # 设置为训练模式以打印调试信息
    hccl.train()
    
    print(f"   目标HU均值: {hccl.target_mean}")
    print(f"   HU容忍范围: ±{hccl.tolerance}")
    print(f"   HU有效范围: [{hccl.hu_low}, {hccl.hu_high}]")
    print(f"   损失权重: {hccl.weight}")
    
    # 计算损失
    print(f"\n4. 计算损失 (训练模式):")
    print("-" * 50)
    
    loss = hccl(prob_batch, ct_batch)
    
    print("-" * 50)
    print(f"   最终加权损失: {loss.item():.6f}")
    
    # 手动验证计算结果
    print(f"\n5. 手动验证:")
    
    with torch.no_grad():
        binary_masks = (prob_batch > 0.5).float()
        
        for i in range(batch_size):
            mask = binary_masks[i:i+1]
            ct = ct_batch[i:i+1]
            
            masked_ct = ct * mask
            num_voxels = torch.sum(mask)
            
            if num_voxels > 0:
                mean_hu = torch.sum(masked_ct) / num_voxels
                
                in_range_mask = (ct >= hccl.hu_low) & (ct <= hccl.hu_high)
                valid_voxels = torch.sum(mask * in_range_mask.float())
                validity_ratio = valid_voxels / num_voxels
                
                print(f"   样本{i}:")
                print(f"     预测体素数: {num_voxels.item():.0f}")
                print(f"     平均HU值: {mean_hu.item():.1f}")
                print(f"     有效比例: {validity_ratio.item():.3f}")
                
                # 计算各项损失
                target = torch.tensor([hccl.target_mean], device=ct.device, dtype=mean_hu.dtype)
                mean_hu_expanded = mean_hu.view(1)  # 修复：确保形状一致
                huber = F.smooth_l1_loss(mean_hu_expanded, target)
                diff = torch.abs(mean_hu - hccl.target_mean)
                tolerance = torch.max(diff - hccl.tolerance, torch.zeros_like(diff))
                validity = 1.0 - validity_ratio
                
                sample_loss = huber + 0.3 * tolerance + 0.2 * validity
                weighted_loss = hccl.weight * sample_loss / batch_size
                
                print(f"     Huber损失: {huber.item():.4f}")
                print(f"     容忍损失: {tolerance.item():.4f}")
                print(f"     有效性惩罚: {validity.item():.4f}")
                print(f"     样本贡献: {weighted_loss.item():.6f}")
    
    # 测试不同HU值的情况
    print(f"\n6. 边界情况测试:")
    
    # 情况1: 空预测（没有EAT体素）
    empty_prob = torch.zeros(1, 1, *spatial_size)
    empty_loss = hccl(empty_prob, ct_batch[:1])
    print(f"   空预测损失: {empty_loss.item():.6f}")
    
    # 情况2: 完美预测（所有体素都在有效范围内）
    perfect_ct = torch.ones(1, 1, *spatial_size) * -110
    perfect_prob = torch.ones(1, 1, *spatial_size) * 0.9
    perfect_loss = hccl(perfect_prob, perfect_ct)
    print(f"   完美预测损失: {perfect_loss.item():.6f}")
    
    # 情况3: 完全错误的HU值
    wrong_ct = torch.ones(1, 1, *spatial_size) * 50  # 心肌的HU值
    wrong_prob = torch.ones(1, 1, *spatial_size) * 0.9
    wrong_loss = hccl(wrong_prob, wrong_ct)
    print(f"   错误HU值损失: {wrong_loss.item():.6f}")
    
    # 与其他损失函数的兼容性测试
    print(f"\n7. 与其他损失函数结合测试:")
    
    # 模拟标准分割损失
    dice_loss_value = 0.2  # 模拟Dice损失
    ce_loss_value = 0.1    # 模拟交叉熵损失
    
    # 修复：先将Tensor转换为Python数值再进行计算和格式化
    loss_value = loss.item()
    total_loss_value = dice_loss_value + ce_loss_value + loss_value
    hccl_ratio = (loss_value / total_loss_value * 100) if total_loss_value > 0 else 0
    
    print(f"   Dice损失: {dice_loss_value:.4f}")
    print(f"   CE损失: {ce_loss_value:.4f}")
    print(f"   HCCL损失: {loss_value:.4f}")
    print(f"   总损失: {total_loss_value:.4f}")
    print(f"   HCCL占总损失比例: {hccl_ratio:.1f}%")
    
    print(f"\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return hccl, prob_batch, ct_batch, loss


if __name__ == "__main__":
    # 运行测试
    hccl_module, preds, cts, test_loss = test_hccl_module()
    
    # 保存测试结果示例
    print(f"\n模块信息:")
    print(f"   模块类型: {type(hccl_module).__name__}")
    print(f"   参数量: {sum(p.numel() for p in hccl_module.parameters())}")
    print(f"   测试损失值: {test_loss.item():.6f}")
    
    # 使用示例
    print(f"\n使用示例:")
    print("""# 在您的训练代码中:
# 1. 初始化HCCL
hccl_loss = HUConsistencyLoss(loss_weight=0.1)

# 2. 在训练循环中计算总损失
for batch in dataloader:
    ct_images, gt_masks = batch  # ct_images应为原始HU值
    preds = model(ct_images)
    
    # 计算各种损失
    dice = dice_loss(preds, gt_masks)
    ce = ce_loss(preds, gt_masks)
    hu_consistency = hccl_loss(preds, ct_images)  # 注意传入原始CT
    
    total_loss = dice + ce + hu_consistency
    
    # 反向传播和优化...
    """)
