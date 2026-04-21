import torch
import torch.nn as nn
import numpy as np

class HUDistributionAgreement:
    """
    HU分布吻合度 (HDA)
    衡量预测EAT区域与真实EAT区域的HU值分布相似度
    
    设计原理:
    1. EAT在CT上有明确的HU范围[-190, -30]
    2. 好的分割应该在HU值分布上也与金标准一致
    3. 提供超越空间重叠度的组织特性一致性评估
    """
    
    @staticmethod
    def compute_hda(pred_mask, gt_mask, ct_image, hu_range=(-200, 0), num_bins=20):
        """
        计算HU分布吻合度
        
        Args:
            pred_mask: 预测的二值分割图 [B, 1, D, H, W] 或 [D, H, W]
            gt_mask: 真实的分割图 [B, 1, D, H, W] 或 [D, H, W]
            ct_image: 原始CT图像（HU值）[B, 1, D, H, W] 或 [D, H, W]
            hu_range: HU值范围用于计算直方图
            num_bins: 直方图bin的数量
            
        Returns:
            hda_score: Bhattacharyya系数，介于0-1之间，越高表示HU分布越一致
        """
        # 确保输入是torch张量
        if not isinstance(pred_mask, torch.Tensor):
            pred_mask = torch.tensor(pred_mask)
        if not isinstance(gt_mask, torch.Tensor):
            gt_mask = torch.tensor(gt_mask)
        if not isinstance(ct_image, torch.Tensor):
            ct_image = torch.tensor(ct_image)
        
        # 处理批次维度
        if pred_mask.dim() == 5:  # [B, 1, D, H, W]
            batch_size = pred_mask.shape[0]  # 修复：获取批次大小的整数
            scores = []
            
            for i in range(batch_size):
                score = HUDistributionAgreement._compute_single(
                    pred_mask[i, 0] if pred_mask.shape[1] == 1 else pred_mask[i],
                    gt_mask[i, 0] if gt_mask.shape[1] == 1 else gt_mask[i],
                    ct_image[i, 0] if ct_image.shape[1] == 1 else ct_image[i],
                    hu_range, num_bins
                )
                scores.append(score)
            
            return torch.tensor(scores).mean()
        else:
            return HUDistributionAgreement._compute_single(
                pred_mask, gt_mask, ct_image, hu_range, num_bins
            )
    
    @staticmethod
    def _compute_single(pred_mask, gt_mask, ct_image, hu_range, num_bins):
        """计算单个样本的HDA"""
        # 确保是二值掩码
        if pred_mask.max() > 1.0 or pred_mask.min() < 0.0:
            pred_mask = (pred_mask > 0.5).float()
        if gt_mask.max() > 1.0 or gt_mask.min() < 0.0:
            gt_mask = (gt_mask > 0.5).float()
        
        # 提取两个mask区域内对应的CT值
        pred_hu_values = ct_image[pred_mask > 0.5]
        gt_hu_values = ct_image[gt_mask > 0.5]
        
        # 如果没有足够的数据点，返回0
        if len(pred_hu_values) < 10 or len(gt_hu_values) < 10:
            return torch.tensor(0.0)
        
        # 定义HU值范围
        hu_min, hu_max = hu_range
        
        # 计算直方图
        pred_hist = torch.histc(pred_hu_values, bins=num_bins, min=hu_min, max=hu_max)
        gt_hist = torch.histc(gt_hu_values, bins=num_bins, min=hu_min, max=hu_max)
        
        # 归一化直方图
        pred_hist = pred_hist / (pred_hist.sum() + 1e-6)
        gt_hist = gt_hist / (gt_hist.sum() + 1e-6)
        
        # 计算Bhattacharyya系数 (BC) 作为分布相似度度量
        bc = torch.sum(torch.sqrt(pred_hist * gt_hist))
        
        return bc
    
    @staticmethod
    def analyze_hu_distributions(pred_mask, gt_mask, ct_image, debug=False):
        """
        分析HU分布的详细统计信息
        
        Args:
            pred_mask: 预测的二值分割图
            gt_mask: 真实的分割图
            ct_image: 原始CT图像（HU值）
            debug: 是否打印调试信息
            
        Returns:
            stats: 包含各种统计信息的字典
        """
        # 确保是二值掩码
        pred_mask_binary = (pred_mask > 0.5).float()
        gt_mask_binary = (gt_mask > 0.5).float()
        
        # 提取HU值
        pred_hu = ct_image[pred_mask_binary > 0.5]
        gt_hu = ct_image[gt_mask_binary > 0.5]
        
        stats = {
            'pred_count': len(pred_hu),
            'gt_count': len(gt_hu),
            'hda_score': None,
            'pred_stats': {},
            'gt_stats': {}
        }
        
        if len(pred_hu) > 0:
            pred_hu_np = pred_hu.cpu().numpy() if torch.is_tensor(pred_hu) else pred_hu
            stats['pred_stats'] = {
                'mean': float(np.mean(pred_hu_np)),
                'std': float(np.std(pred_hu_np)),
                'min': float(np.min(pred_hu_np)),
                'max': float(np.max(pred_hu_np)),
                'median': float(np.median(pred_hu_np)),
                'q1': float(np.percentile(pred_hu_np, 25)),
                'q3': float(np.percentile(pred_hu_np, 75))
            }
        
        if len(gt_hu) > 0:
            gt_hu_np = gt_hu.cpu().numpy() if torch.is_tensor(gt_hu) else gt_hu
            stats['gt_stats'] = {
                'mean': float(np.mean(gt_hu_np)),
                'std': float(np.std(gt_hu_np)),
                'min': float(np.min(gt_hu_np)),
                'max': float(np.max(gt_hu_np)),
                'median': float(np.median(gt_hu_np)),
                'q1': float(np.percentile(gt_hu_np, 25)),
                'q3': float(np.percentile(gt_hu_np, 75))
            }
        
        # 计算HDA分数
        if len(pred_hu) > 10 and len(gt_hu) > 10:
            stats['hda_score'] = float(HUDistributionAgreement.compute_hda(
                pred_mask, gt_mask, ct_image
            ))
        
        # 打印调试信息
        if debug:
            print("\n[HDA分析]")
            print(f"预测EAT体素数: {stats['pred_count']}")
            print(f"真实EAT体素数: {stats['gt_count']}")
            
            if stats['pred_stats']:
                print(f"\n预测EAT HU统计:")
                print(f"  均值: {stats['pred_stats']['mean']:.1f}")
                print(f"  标准差: {stats['pred_stats']['std']:.1f}")
                print(f"  范围: [{stats['pred_stats']['min']:.1f}, {stats['pred_stats']['max']:.1f}]")
                print(f"  中位数: {stats['pred_stats']['median']:.1f}")
                print(f"  Q1-Q3: [{stats['pred_stats']['q1']:.1f}, {stats['pred_stats']['q3']:.1f}]")
            
            if stats['gt_stats']:
                print(f"\n真实EAT HU统计:")
                print(f"  均值: {stats['gt_stats']['mean']:.1f}")
                print(f"  标准差: {stats['gt_stats']['std']:.1f}")
                print(f"  范围: [{stats['gt_stats']['min']:.1f}, {stats['gt_stats']['max']:.1f}]")
                print(f"  中位数: {stats['gt_stats']['median']:.1f}")
                print(f"  Q1-Q3: [{stats['gt_stats']['q1']:.1f}, {stats['gt_stats']['q3']:.1f}]")
            
            if stats['hda_score'] is not None:
                print(f"\nHDA分数: {stats['hda_score']:.4f}")
                if stats['hda_score'] > 0.9:
                    print("  → HU分布一致性: 优秀")
                elif stats['hda_score'] > 0.8:
                    print("  → HU分布一致性: 良好")
                elif stats['hda_score'] > 0.7:
                    print("  → HU分布一致性: 中等")
                else:
                    print("  → HU分布一致性: 较差")
        
        return stats


def test_hda_module():
    """测试HDA模块的功能"""
    print("=" * 60)
    print("HU分布吻合度 (HDA) 模块测试")
    print("=" * 60)
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
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
    pred_masks = []
    gt_masks = []
    
    # 样本0: "完美"的分割 - HU分布几乎相同
    ct0 = torch.randn(1, 1, *spatial_size) * 30 - 110  # 均值-110，标准差30
    gt0 = torch.zeros(1, 1, *spatial_size)
    gt0[:, :, 10:22, 10:22, 5:11] = 1
    pred0 = gt0.clone() * 0.95  # 轻微差异
    ct_images.append(ct0)
    pred_masks.append(pred0)
    gt_masks.append(gt0)
    
    # 样本1: "一般"的分割 - HU分布有差异
    ct1 = torch.randn(1, 1, *spatial_size) * 40 - 120  # 均值-120，标准差40
    gt1 = torch.zeros(1, 1, *spatial_size)
    gt1[:, :, 12:24, 12:24, 6:12] = 1
    pred1 = torch.zeros(1, 1, *spatial_size)
    pred1[:, :, 11:23, 11:23, 5:11] = 1  # 偏移了一点
    ct_images.append(ct1)
    pred_masks.append(pred1)
    gt_masks.append(gt1)
    
    # 样本2: "错误"的分割 - HU分布完全不同
    ct2 = torch.randn(1, 1, *spatial_size) * 30 - 130  # 均值-130，标准差30
    gt2 = torch.zeros(1, 1, *spatial_size)
    gt2[:, :, 14:26, 14:26, 7:13] = 1
    pred2 = torch.zeros(1, 1, *spatial_size)
    pred2[:, :, 5:17, 5:17, 2:8] = 1  # 完全不同的位置
    ct_images.append(ct2)
    pred_masks.append(pred2)
    gt_masks.append(gt2)
    
    # 合并批次
    ct_batch = torch.cat(ct_images, dim=0)
    pred_batch = torch.cat(pred_masks, dim=0)
    gt_batch = torch.cat(gt_masks, dim=0)
    
    print(f"   CT图像形状: {ct_batch.shape}")
    print(f"   预测掩码形状: {pred_batch.shape}")
    print(f"   真实掩码形状: {gt_batch.shape}")
    print(f"   CT值范围: [{ct_batch.min():.1f}, {ct_batch.max():.1f}] HU")
    
    # 测试基本HDA计算
    print(f"\n3. 计算HDA分数:")
    print("-" * 50)
    
    for i in range(batch_size):
        hda_score = HUDistributionAgreement.compute_hda(
            pred_batch[i:i+1], 
            gt_batch[i:i+1], 
            ct_batch[i:i+1]
        )
        print(f"   样本{i} HDA分数: {hda_score.item():.4f}")
    
    # 批量计算
    batch_hda = HUDistributionAgreement.compute_hda(pred_batch, gt_batch, ct_batch)
    print(f"\n   批量平均HDA: {batch_hda.item():.4f}")
    
    # 详细分析
    print(f"\n4. 详细分析 (样本0):")
    print("-" * 50)
    
    stats = HUDistributionAgreement.analyze_hu_distributions(
        pred_batch[0], 
        gt_batch[0], 
        ct_batch[0], 
        debug=True
    )
    
    # 测试边界情况
    print(f"\n5. 边界情况测试:")
    print("-" * 50)
    
    # 情况1: 完美匹配
    perfect_ct = torch.ones(1, 1, *spatial_size) * -110
    perfect_pred = torch.ones(1, 1, *spatial_size)
    perfect_gt = torch.ones(1, 1, *spatial_size)
    perfect_hda = HUDistributionAgreement.compute_hda(perfect_pred, perfect_gt, perfect_ct)
    print(f"   完美匹配HDA: {perfect_hda.item():.4f}")
    
    # 情况2: 空预测
    empty_pred = torch.zeros(1, 1, *spatial_size)
    empty_hda = HUDistributionAgreement.compute_hda(empty_pred, gt_batch[0:1], ct_batch[0:1])
    print(f"   空预测HDA: {empty_hda.item():.4f}")
    
    # 情况3: 无真实标注
    no_gt = torch.zeros(1, 1, *spatial_size)
    no_gt_hda = HUDistributionAgreement.compute_hda(pred_batch[0:1], no_gt, ct_batch[0:1])
    print(f"   无真实标注HDA: {no_gt_hda.item():.4f}")
    
    # 情况4: 完全不同的HU分布
    ct_warm = torch.ones(1, 1, *spatial_size) * -50  # 较暖的脂肪
    ct_cold = torch.ones(1, 1, *spatial_size) * -160  # 较冷的脂肪
    pred_warm = torch.ones(1, 1, *spatial_size)
    gt_cold = torch.ones(1, 1, *spatial_size)
    diff_hda = HUDistributionAgreement.compute_hda(pred_warm, gt_cold, ct_warm)
    print(f"   不同HU分布HDA: {diff_hda.item():.4f}")
    
    # 与其他指标的对比
    print(f"\n6. 与其他指标的关系:")
    print("-" * 50)
    
    # 模拟Dice系数（空间重叠度）
    def compute_dice(pred, gt):
        intersection = torch.sum(pred * gt)
        union = torch.sum(pred) + torch.sum(gt)
        return 2.0 * intersection / (union + 1e-6)
    
    for i in range(batch_size):
        dice = compute_dice(pred_batch[i], gt_batch[i])
        hda = HUDistributionAgreement.compute_hda(
            pred_batch[i:i+1], 
            gt_batch[i:i+1], 
            ct_batch[i:i+1]
        )
        print(f"   样本{i}: Dice={dice.item():.4f}, HDA={hda.item():.4f}")
    
    print(f"\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    return HUDistributionAgreement, pred_batch, gt_batch, ct_batch, batch_hda


if __name__ == "__main__":
    # 运行测试
    hda_class, preds, gts, cts, avg_hda = test_hda_module()
    
    # 保存测试结果示例
    print(f"\n模块信息:")
    print(f"   模块类型: {hda_class.__name__}")
    print(f"   平均HDA分数: {avg_hda.item():.4f}")
    
    # 使用示例
    print(f"\n使用示例:")
    print("""# 在您的评估代码中:
    # 1. 导入HDA模块
    from your_module import HUDistributionAgreement

    # 2. 在评估循环中计算HDA
    for batch in test_dataloader:
        ct_images, gt_masks = batch  # ct_images应为原始HU值
        preds = model(ct_images)
        
        # 计算传统指标
        dice = dice_score(preds, gt_masks)
        hd95 = hausdorff_distance(preds, gt_masks)
        
        # 计算HDA（新增的组织特性一致性指标）
        hda = HUDistributionAgreement.compute_hda(preds, gt_masks, ct_images)
        
        # 详细分析（可选）
        stats = HUDistributionAgreement.analyze_hu_distributions(
            preds[0], gt_masks[0], ct_images[0], debug=True
        )
        
        print(f"Dice: {dice:.4f}, HD95: {hd95:.2f}, HDA: {hda:.4f}")

    # 3. 在论文中报告结果时:
    #    - 除了DSC、HD95等空间指标外，新增HDA指标
    #    - 说明HDA反映了分割结果在组织特性（HU值分布）上的准确性
    #    - 高HDA表明模型不仅分割了正确的区域，还分割了正确的组织类型
    """
          )
