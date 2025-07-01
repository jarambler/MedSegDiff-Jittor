import os
import sys
sys.path.append("../")
sys.path.append("./")
import argparse
import numpy as np
import jittor as jt
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm

from medsegdiff import (
    initialize_jittor_for_cuda,
    ensure_float32_tensor,
    convert_numpy_to_jittor_float32
)


def create_table(headers, rows):
    """创建一个表格"""
    # 计算列宽
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # 创建格式字符串
    format_str = " | ".join(f"{{:<{w}}}" for w in col_widths)

    # 创建表格
    table_lines = []
    table_lines.append(format_str.format(*headers))
    table_lines.append("-" * (sum(col_widths) + 3 * (len(headers) - 1)))

    for row in rows:
        table_lines.append(format_str.format(*[str(cell) for cell in row]))

    return "\n".join(table_lines)


def dice_coefficient(pred, target, smooth=1e-6):
    """计算Dice系数"""
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    """计算IoU（交并比）"""
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def evaluate_predictions(pred_dir, gt_dir, threshold=0.5, dataset='brats2021'):
    """评估预测结果与ground truth"""
    pred_files = list(Path(pred_dir).glob("*_fused.png"))

    if not pred_files:
        print(f"No prediction files found in {pred_dir}")
        return None, None

    dice_scores = []
    iou_scores = []

    for pred_file in tqdm(pred_files, desc="Evaluating"):
        # 找到相应的ground truth
        sample_name = pred_file.stem.replace("_fused", "")

        gt_file = None

        if dataset == 'brats2021':
            if '_t1.nii' in sample_name:
                base_name = sample_name.replace('_t1.nii', '')
            elif '_t2.nii' in sample_name:
                base_name = sample_name.replace('_t2.nii', '')
            elif '_flair.nii' in sample_name:
                base_name = sample_name.replace('_flair.nii', '')
            elif '_t1ce.nii' in sample_name:
                base_name = sample_name.replace('_t1ce.nii', '')
            else:
                base_name = sample_name.replace('.nii', '').replace('_t1', '').replace('_t2', '').replace('_flair', '').replace('_t1ce', '')

            gt_patterns = [
                f"{base_name}/{base_name}_seg.nii.gz",
            ]
        else:
            # 其他数据集
            gt_patterns = [
                f"{sample_name}_mask.png",
                f"{sample_name}_gt.png",
                f"{sample_name}.png",
                f"{sample_name}_seg.png"
            ]

        for pattern in gt_patterns:
            potential_gt = Path(gt_dir) / pattern
            if potential_gt.exists():
                gt_file = potential_gt
                break

        if gt_file is None:
            print(f"Warning: No ground truth found for {sample_name} (base: {base_name if dataset == 'brats2021' else 'N/A'})")
            continue
        
        try:
            # 加载预测结果
            pred_img = Image.open(pred_file).convert('L')
            pred_array = np.array(pred_img).astype(np.float32) / 255.0

            # 加载ground truth（处理不同格式）
            if gt_file.suffix.lower() in ['.nii', '.gz']:
                # 处理BraTS2021的NIfTI文件
                try:
                    import nibabel as nib
                    gt_nii = nib.load(gt_file)
                    gt_data = gt_nii.get_fdata()

                    # 对于BraTS2021，取中间切片或平均切片
                    if len(gt_data.shape) == 3:
                        # 取中间切片
                        middle_slice = gt_data.shape[2] // 2
                        gt_array = gt_data[:, :, middle_slice]
                    else:
                        gt_array = gt_data

                    # 归一化到0-1范围
                    gt_array = gt_array.astype(np.float32)
                    if gt_array.max() > 1:
                        gt_array = gt_array / gt_array.max()

                except ImportError:
                    print(f"Warning: nibabel not available, skipping NIfTI file {gt_file}")
                    continue
                except Exception as e:
                    print(f"Warning: Error loading NIfTI file {gt_file}: {e}")
                    continue
            else:
                # 处理PNG/JPG文件
                gt_img = Image.open(gt_file).convert('L')
                gt_array = np.array(gt_img).astype(np.float32) / 255.0

            # 如果ground truth的形状与预测不匹配，则调整ground truth的形状
            if gt_array.shape != pred_array.shape:
                gt_img_resized = Image.fromarray((gt_array * 255).astype(np.uint8))
                gt_img_resized = gt_img_resized.resize(pred_img.size, Image.NEAREST)
                gt_array = np.array(gt_img_resized).astype(np.float32) / 255.0

            # 应用阈值
            pred_binary = (pred_array > threshold).astype(np.float32)
            gt_binary = (gt_array > threshold).astype(np.float32)

            # 转换为Jittor张量，数据类型为float32
            pred_tensor = convert_numpy_to_jittor_float32(pred_binary)
            gt_tensor = convert_numpy_to_jittor_float32(gt_binary)

            # 确保张量是float32
            pred_tensor = ensure_float32_tensor(pred_tensor)
            gt_tensor = ensure_float32_tensor(gt_tensor)
            
            # 计算指标
            dice = dice_coefficient(pred_tensor, gt_tensor)
            iou = iou_score(pred_tensor, gt_tensor)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            
        except Exception as e:
            print(f"Error processing {pred_file}: {e}")
            continue
    
    if not dice_scores:
        return None, None
    
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    
    return mean_dice, mean_iou


def find_model_results(output_dir, dataset):
    """找到所有模型的结果目录"""
    output_path = Path(output_dir)
    test_results_path = output_path / "test"

    model_dirs = []

    if test_results_path.exists():
        for model_dir in test_results_path.iterdir():
            if model_dir.is_dir():
                if model_dir.name.endswith(f"_{dataset}"):
                    model_name = model_dir.name.replace(f"_{dataset}", "")
                    model_dirs.append((model_name, model_dir))

    return model_dirs


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MedSegDiff Models')

    # 输入/输出路径
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory containing test results')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory containing ground truth masks')

    # 数据集配置
    parser.add_argument('--dataset', type=str, default='brats2021',
                        choices=['brats2021', 'refuge2', 'ddti'],
                        help='Dataset name for evaluation')

    # 评估参数
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation (default: 0.5)')

    # 输出选项
    parser.add_argument('--save_results', type=str, default='evaluation_results.json',
                        help='File to save evaluation results')
    parser.add_argument('--results_format', type=str, default='json',
                        choices=['json', 'csv', 'both'],
                        help='Format for saving results (default: json)')

    return parser.parse_args()


def print_evaluation_summary(results, dataset_name):
    """打印数据集的评估结果"""
    print(f"\n{dataset_name.upper()} EVALUATION RESULTS")
    print("=" * 60)

    # 分离基础模型和消融研究
    base_models = []
    ablation_models = []

    for model_name in results.keys():
        if any(suffix in model_name for suffix in ['no_ff_parser', 'no_dynamic_encoding']):
            ablation_models.append(model_name)
        else:
            base_models.append(model_name)

    # 排序模型
    model_order = ['MedSegDiff-S', 'MedSegDiff-B', 'MedSegDiff-L', 'MedSegDiff++']
    base_models.sort(key=lambda x: model_order.index(x.split('_')[0]) if x.split('_')[0] in model_order else 999)
    ablation_models.sort()

    # 为基础模型创建表格
    if base_models:
        print("\nBase Models Performance")
        headers = ["Model", "Dice", "IoU", "Samples"]
        rows = []

        for model_name in base_models:
            result = results[model_name]
            if result['dice'] is not None:
                dice_str = f"{result['dice']:.4f}"
                iou_str = f"{result['iou']:.4f}"
            else:
                dice_str = "FAILED"
                iou_str = "FAILED"

            rows.append([model_name, dice_str, iou_str, result['num_samples']])

        print(create_table(headers, rows))

    # 为消融研究创建表格
    if ablation_models:
        print("\nAblation Study Results")
        headers = ["Model", "Dice", "IoU", "Samples"]
        rows = []

        for model_name in ablation_models:
            result = results[model_name]
            if result['dice'] is not None:
                dice_str = f"{result['dice']:.4f}"
                iou_str = f"{result['iou']:.4f}"
            else:
                dice_str = "FAILED"
                iou_str = "FAILED"

            rows.append([model_name, dice_str, iou_str, result['num_samples']])

        print(create_table(headers, rows))


def main():
    args = parse_args()

    # 使用CUDA和float32
    initialize_jittor_for_cuda()

    print("MedSegDiff Model Evaluation")
    print("=" * 50)

    # 找到所有模型的预测结果目录
    model_dirs = find_model_results(args.output_dir, args.dataset)

    if not model_dirs:
        print(f"No model results found in {args.output_dir}/test")
        print(f"Available directories in {args.output_dir}/test:")
        test_path = Path(args.output_dir) / "test"
        if test_path.exists():
            for d in test_path.iterdir():
                if d.is_dir():
                    print(f"  - {d.name}")
        return

    # 验证ground truth目录
    gt_path = Path(args.gt_dir)
    if not gt_path.exists():
        print(f"Error: Ground truth directory not found: {args.gt_dir}")       
        return

    print(f"Found {len(model_dirs)} model result directories")
    print(f"Test results directory: {args.output_dir}/test")
    print(f"Available model directories:")
    for _, result_dir in model_dirs:
        print(f"  - {result_dir}")
    print(f"Ground truth directory: {args.gt_dir}")
    print(f"Dataset: {args.dataset}")
    print(f"Threshold: {args.threshold}")

    # 评估每个模型
    results = {}

    for model_name, result_dir in model_dirs:
        print(f"\nEvaluating {model_name}...")

        dice, iou = evaluate_predictions(result_dir, args.gt_dir, args.threshold, args.dataset)

        if dice is not None and iou is not None:
            results[model_name] = {
                'dice': dice,
                'iou': iou,
                'num_samples': len(list(result_dir.glob("*_fused.png")))
            }
            print(f"  Dice: {dice:.4f}, IoU: {iou:.4f}")
        else:
            results[model_name] = {
                'dice': None,
                'iou': None,
                'num_samples': 0
            }
            print(f"  Evaluation failed")

    # 打印评估结果
    print_evaluation_summary(results, args.dataset)
    
    # 找到表现最好的模型
    valid_results = {k: v for k, v in results.items() if v['dice'] is not None}
    if valid_results:
        best_model = max(valid_results.keys(), key=lambda k: valid_results[k]['dice'])
        best_dice = valid_results[best_model]['dice']
        best_iou = valid_results[best_model]['iou']
        
        print(f"\nBest Model: {best_model}")
        print(f"   Dice: {best_dice:.4f}, IoU: {best_iou:.4f}")
    
    # 将结果保存为JSON
    if args.save_results:
        results_with_metadata = {
            'dataset': args.dataset,
            'threshold': args.threshold,
            'evaluation_models': [model_name for model_name, _ in model_dirs],
            'gt_dir': args.gt_dir,
            'results': results
        }
        
        with open(args.save_results, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\nResults saved to {args.save_results}")
    
    print(f"\nEvaluation completed!")


if __name__ == "__main__":
    main()
