import os
import sys
sys.path.append("../")
sys.path.append("./")
import argparse
import numpy as np
import jittor as jt
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from medsegdiff import Unet, GaussianDiffusion
from dataset_prepare import BraTsDataset, RefugeDataset, DDTIDataset
from jittor import transform

from medsegdiff import (
    initialize_jittor_for_cuda,
    ensure_float32_tensor,
    convert_numpy_to_jittor_float32,
    setup_model_float32,
    setup_diffusion_float32
)


def staple_algorithm(segmentations, threshold=0.5):
    """STAPLE算法用于集成融合"""
    if len(segmentations) == 0:
        return None
    
    if isinstance(segmentations[0], jt.Var):
        segmentations = [seg.numpy() for seg in segmentations]
    
    # 堆叠所有segmentations
    stacked = np.stack(segmentations, axis=0)
    
    # 转换为二进制
    binary_segs = (stacked > threshold).astype(np.float32)
    
    # 简单多数投票
    fused = np.mean(binary_segs, axis=0)
    
    return fused


def get_model_config(variant, dataset='brats2021'):
    """获取测试模型配置"""
    if dataset in ['refuge2', 'ddti', 'brats2021']:
        configs = {
            'MedSegDiff-S': {
                'dim_mults': (1, 2, 4, 8),
                'base_dim': 64,
                'ensemble_size': 25,  # 集成次数
            },
            'MedSegDiff-B': {
                'dim_mults': (1, 2, 4, 8, 16),
                'base_dim': 64,
                'ensemble_size': 25,
            },
            'MedSegDiff-L': {
                'dim_mults': (1, 2, 4, 8, 16, 32),
                'base_dim': 64,
                'ensemble_size': 25,
            },
            'MedSegDiff++': {
                'dim_mults': (1, 2, 4, 8, 16, 32),
                'base_dim': 64,
                'ensemble_size': 25,
            }
        }

    return configs[variant]


def create_model(variant, image_size, input_channels=4, mask_channels=1, self_condition=False,
                disable_ff_parser=False, disable_dynamic_encoding=False, dataset='brats2021'):
    """根据模型变体创建模型"""

    model_config = get_model_config(variant, dataset)
    
    skip_connect_condition_famps = not disable_dynamic_encoding
    dynamic_ff_parser_attn_map = not disable_ff_parser
    
    num_downsample = len(model_config['dim_mults']) - 1
    min_feature_size = image_size // (2 ** num_downsample)
    
    if variant == 'MedSegDiff++':
        patch_size = 2
        while patch_size <= min_feature_size and min_feature_size % patch_size != 0:
            patch_size += 1
        if patch_size > min_feature_size:
            patch_size = min_feature_size

        conditioning_kwargs = {
            'heads': 8,
            'dim_head': 64,
            'depth': 6,
            'patch_size': patch_size
        }
        full_self_attn = tuple([False] * (len(model_config['dim_mults']) - 1) + [True])
        mid_transformer_depth = 1
        attn_heads = 8
        attn_dim_head = 64

    elif variant == 'MedSegDiff-L':
        patch_size = 2
        while patch_size <= min_feature_size and min_feature_size % patch_size != 0:
            patch_size += 1
        if patch_size > min_feature_size:
            patch_size = min_feature_size

        conditioning_kwargs = {
            'heads': 6,
            'dim_head': 48,
            'depth': 4,
            'patch_size': patch_size
        }
        full_self_attn = tuple([False] * (len(model_config['dim_mults']) - 1) + [True])
        mid_transformer_depth = 1
        attn_heads = 6
        attn_dim_head = 48

    elif variant == 'MedSegDiff-B':
        patch_size = 2
        while patch_size <= min_feature_size and min_feature_size % patch_size != 0:
            patch_size += 1
        if patch_size > min_feature_size:
            patch_size = min_feature_size

        conditioning_kwargs = {
            'heads': 4,
            'dim_head': 32,
            'depth': 3,
            'patch_size': patch_size
        }
        full_self_attn = tuple([False] * (len(model_config['dim_mults']) - 1) + [True])
        mid_transformer_depth = 1
        attn_heads = 4
        attn_dim_head = 32

    else:
        patch_size = 2
        while patch_size <= min_feature_size and min_feature_size % patch_size != 0:
            patch_size += 1
        if patch_size > min_feature_size:
            patch_size = min_feature_size

        conditioning_kwargs = {
            'heads': 2,
            'dim_head': 16,
            'depth': 2,
            'patch_size': patch_size
        }
        full_self_attn = tuple([False] * (len(model_config['dim_mults']) - 1) + [True])
        mid_transformer_depth = 1
        attn_heads = 2
        attn_dim_head = 16

    model = Unet(
        dim=model_config['base_dim'],
        image_size=image_size,
        dim_mults=model_config['dim_mults'],
        full_self_attn=full_self_attn,
        attn_heads=attn_heads,
        attn_dim_head=attn_dim_head,
        mask_channels=mask_channels,
        input_img_channels=input_channels,
        self_condition=self_condition,
        skip_connect_condition_famps=skip_connect_condition_famps,
        dynamic_ff_parser_attn_map=dynamic_ff_parser_attn_map,
        mid_transformer_depth=mid_transformer_depth,
        conditioning_kwargs=conditioning_kwargs
    )
    
    return model


def load_test_data(data_path, dataset, image_size):
    """Load test dataset"""
    
    if dataset == 'brats2021':
        transform_list = [transform.Resize((image_size, image_size))]
        transform_test = transform.Compose(transform_list)
        test_dataset = BraTsDataset(data_path, transform=transform_test, training=False, test_flag=True)
        input_channels = 4
        mask_channels = 1
    elif dataset == 'refuge2':
        transform_list = [transform.Resize((image_size, image_size)), transform.ToTensor()]
        transform_test = transform.Compose(transform_list)
        test_dataset = RefugeDataset(data_path, transform=transform_test, training=False)
        input_channels = 3  # RGB图像
        mask_channels = 1
    elif dataset == 'ddti':
        transform_list = [transform.Resize((image_size, image_size)), transform.ToTensor()]
        transform_test = transform.Compose(transform_list)
        test_dataset = DDTIDataset(data_path, transform=transform_test, training=False)
        input_channels = 3  # RGB图像
        mask_channels = 1
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    test_loader = test_dataset.set_attrs(batch_size=1, shuffle=False)
    
    return test_loader, input_channels, mask_channels


def generate_predictions_for_model(model_path, variant, dataset, data_path, output_dir,
                                 ensemble_size, image_size, disable_ff_parser=False, disable_dynamic_encoding=False,
                                 calculate_loss=True, self_condition=False):
    """使用所选模型进行预测，并计算损失"""

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return False, None

    print(f"Generating predictions for {variant}")

    # 加载测试数据
    test_loader, input_channels, mask_channels = load_test_data(data_path, dataset, image_size)

    # 加载检查点以获取模型配置
    try:
        checkpoint = jt.load(model_path)

        # 获取训练参数以确定模型配置
        train_self_condition = self_condition
        train_dataset = dataset
        train_input_channels = input_channels

        if 'args' in checkpoint:
            train_args = checkpoint['args']
            train_dataset = train_args.get('dataset', dataset)
            train_self_condition = train_args.get('self_condition', self_condition)
            print(f"Found training config: dataset={train_dataset}, self_condition={train_self_condition}")

        # 从模型权重获取输入通道
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']

            # 检查cond_init_conv的输入通道
            if 'cond_init_conv.weight' in state_dict:
                cond_conv_weight = state_dict['cond_init_conv.weight']
                if hasattr(cond_conv_weight, 'shape') and len(cond_conv_weight.shape) >= 2:
                    train_input_channels = cond_conv_weight.shape[1]
                    print(f"Detected actual input channels from cond_init_conv: {train_input_channels}")

            # 检查init_con的掩码通道
            if 'init_conv.weight' in state_dict:
                init_conv_weight = state_dict['init_conv.weight']
                if hasattr(init_conv_weight, 'shape') and len(init_conv_weight.shape) >= 2:
                    expected_mask_channels = init_conv_weight.shape[1]

                    if expected_mask_channels == mask_channels * 2:
                        train_self_condition = True
                        print(f"Detected self_condition=True from init_conv channels: {expected_mask_channels}")
                    elif expected_mask_channels == mask_channels:
                        train_self_condition = False
                        print(f"Detected self_condition=False from init_conv channels: {expected_mask_channels}")

    except Exception as e:
        print(f"Warning: Could not load checkpoint for config check: {e}")
        train_input_channels = input_channels
        train_dataset = dataset
        train_self_condition = self_condition

    # 使用训练配置创建模型
    model = create_model(variant, image_size=image_size, input_channels=train_input_channels, mask_channels=mask_channels,
                        disable_ff_parser=disable_ff_parser, disable_dynamic_encoding=disable_dynamic_encoding,
                        dataset=train_dataset, self_condition=train_self_condition)

    print(f"Model created with: input_channels={train_input_channels}, self_condition={train_self_condition}")

    # 存储实际输入通道以进行数据处理
    actual_input_channels = train_input_channels

    # 加载checkpoint
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded successfully (input_channels: {actual_input_channels})")

    except Exception as e:
        print(f"Error loading model: {e}")
        return False, None

    model = setup_model_float32(model)

    # 创建扩散模型
    diffusion = GaussianDiffusion(model, timesteps=100, sampling_timesteps=100, objective='pred_noise')

    diffusion = setup_diffusion_float32(diffusion)
    os.makedirs(output_dir, exist_ok=True)

    total_loss = 0.0
    num_samples = 0
    losses = []

    # 测试
    for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Generating predictions")):
        if len(batch_data) == 2:
            img, gt_mask = batch_data
            filename = f"sample_{batch_idx:04d}"
        elif len(batch_data) == 3:
            img, gt_mask, path = batch_data
            if isinstance(path, (list, tuple)):
                filename = Path(path[0]).stem
            else:
                filename = Path(path).stem
        else:
            continue

        img = convert_numpy_to_jittor_float32(img)
        gt_mask = convert_numpy_to_jittor_float32(gt_mask)

        img = ensure_float32_tensor(img)
        gt_mask = ensure_float32_tensor(gt_mask)

        if hasattr(img, 'shape') and len(img.shape) >= 3:
            current_channels = img.shape[1] if len(img.shape) == 4 else img.shape[0]

            if current_channels != actual_input_channels:
                print(f"Adjusting image channels: {current_channels} -> {actual_input_channels}")

                if actual_input_channels == 1 and current_channels == 3:
                    # 将RGB转换为灰度
                    if len(img.shape) == 4:  # [B, C, H, W]
                        img = jt.mean(img, dim=1, keepdims=True)  # 平均RGB通道
                    else:  # [C, H, W]
                        img = jt.mean(img, dim=0, keepdims=True)  # 平均RGB通道

                elif actual_input_channels == 2 and current_channels == 3:
                    if len(img.shape) == 4:  # [B, C, H, W]
                        img = img[:, :2, :, :]
                    else:  # [C, H, W]
                        img = img[:2, :, :]

                elif actual_input_channels == 3 and current_channels == 1:
                    if len(img.shape) == 4:  # [B, C, H, W]
                        img = img.repeat(1, 3, 1, 1)
                    else:  # [C, H, W]
                        img = img.repeat(3, 1, 1)

                elif actual_input_channels == 4 and current_channels == 3:
                    if len(img.shape) == 4:  # [B, C, H, W]
                        extra_channel = jt.zeros((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=jt.float32)
                        img = jt.concat([img, extra_channel], dim=1)
                    else:  # [C, H, W]
                        extra_channel = jt.zeros((1, img.shape[1], img.shape[2]), dtype=jt.float32)
                        img = jt.concat([img, extra_channel], dim=0)

                print(f"Adjusted image shape: {img.shape}")

        img = ensure_float32_tensor(img)
        gt_mask = ensure_float32_tensor(gt_mask)

        # 计算损失
        if calculate_loss and gt_mask is not None:
            try:
                with jt.no_grad():
                    if hasattr(gt_mask, 'shape') and len(gt_mask.shape) >= 3:
                        if gt_mask.shape[1] > 1:  # [B, C, H, W]
                            gt_mask = gt_mask[:, 0:1, :, :]  # 取第一个通道
                        elif len(gt_mask.shape) == 3 and gt_mask.shape[0] > 1:  # [C, H, W]
                            gt_mask = gt_mask[0:1, :, :]  # 取第一个通道

                    loss = diffusion(gt_mask, img)
                    loss = ensure_float32_tensor(loss)
                    if not jt.isnan(loss).any() and not jt.isinf(loss).any():
                        total_loss += loss.item()
                        losses.append(loss.item())
                        num_samples += 1
            except Exception as e:
                print(f"Warning: Could not calculate loss for sample {batch_idx}: {e}")

        # 计算集成
        ensemble_masks = []

        for _ in range(ensemble_size):
            with jt.no_grad():
                pred_mask = diffusion.sample(img)
            ensemble_masks.append(pred_mask[0, 0])  # 取第一个批处理项和通道

        # 应用STAPLE融合
        fused_mask = staple_algorithm(ensemble_masks)

        # 保存融合结果
        fused_img = Image.fromarray((fused_mask * 255).astype(np.uint8))
        fused_img.save(output_dir / f"{filename}_fused.png")

    # 计算平均损失
    avg_loss = total_loss / max(num_samples, 1) if num_samples > 0 else None

    print(f"Predictions saved to {output_dir}")
    if calculate_loss and avg_loss is not None:
        print(f"Average test loss: {avg_loss:.6f} (calculated on {num_samples} samples)")

    return True, avg_loss


def find_trained_models(models_dir, args):
    """找到所有训练好的模型检查点并进行选择"""
    models = []
    models_path = Path(models_dir)

    for model_dir in models_path.iterdir():
        if model_dir.is_dir():
            checkpoint_dir = model_dir / "checkpoints"
            if checkpoint_dir.exists():
                if args.best_model:
                    # 首先查找最佳模型
                    best_model = checkpoint_dir / f"{model_dir.name}_best.pkl"
                    if best_model.exists():
                        models.append((model_dir.name, str(best_model)))
                        continue

                # 如果没有找到最佳模型，则查找损失最小的模型
                model_files = []
                for model_file in checkpoint_dir.iterdir():
                    if model_file.is_file() and model_file.name.endswith('.pkl'):
                        # 从文件名中提取损失
                        loss_part = model_file.name.split('_loss_')[1].replace('.pkl', '')
                        loss_value = float(loss_part)
                        model_files.append((model_file.name, str(model_file), loss_value))                

                if model_files:
                    if args.best_model or len(model_files) == 1:
                        # 选择损失最小的模型
                        best_model_info = min(model_files, key=lambda x: x[2])
                        models.append((model_dir.name, best_model_info[1]))
                        print(f"Selected {model_dir.name}: {best_model_info[0]} (loss: {best_model_info[2]:.4f})")
                    else:
                        # 如果没有best_model参数，则添加所有模型
                        for model_name, model_path, _ in model_files:
                            models.append((f"{model_dir.name}_{model_name.replace('.pkl', '')}", model_path))

    return models


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Predictions for MedSegDiff Models')

    # 输入/输出路径
    parser.add_argument('--models_dir', type=str, default='output/train',
                        help='Directory containing trained models')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to test data directory')

    # 数据集配置
    parser.add_argument('--dataset', type=str, default='brats2021',
                        choices=['brats2021', 'refuge2', 'ddti'],
                        help='Dataset name for testing')
    parser.add_argument('--image_size', type=int, default=None,
                        help='Image size for testing (auto-detected if not specified)')

    # 模型选择
    parser.add_argument('--specific_model', type=str, default=None,
                        help='Generate predictions for specific model only')
    parser.add_argument('--best_model', action='store_true',
                        help='Test the best model on the specified dataset')

    # 推理参数
    parser.add_argument('--ensemble_size', type=int, default=None,
                        help='Number of samples for ensemble prediction (default: None)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    
    parser.add_argument('--output_dir', type=str, default='output/test',
                        help='Output directory for saving predictions')

    parser.add_argument('--self_condition', action='store_true',
                        help='Enable self-conditioning during inference')

    return parser.parse_args()


def main():
    args = parse_args()

    # 使用CUDA和float32
    initialize_jittor_for_cuda()

    print("MedSegDiff Prediction Generation")
    print("=" * 50)

    # 找到训练好的模型
    models = find_trained_models(args.models_dir, args)

    if not models:
        print(f"No trained models found in {args.models_dir}")
        return

    if args.specific_model:
        models = [(name, path) for name, path in models if args.specific_model in name]
        if not models:
            print(f"Model {args.specific_model} not found")
            return

    print(f"Found {len(models)} trained models")
    print(f"Test data: {args.data_path}")
    print(f"Dataset: {args.dataset}")

    # 为每个模型生成预测
    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True)

    for model_name, model_path in models:
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}")

        # 解析模型配置
        parts = model_name.split('_')
        variant = parts[0]
        dataset = parts[1] if len(parts) > 1 and parts[1] in ['brats2021', 'refuge2', 'ddti'] else args.dataset

        disable_ff_parser = 'no_ff_parser' in model_name
        disable_dynamic_encoding = 'no_dynamic_encoding' in model_name

        model_config = get_model_config(variant, dataset)
        if args.ensemble_size == None: 
            ensemble_size = model_config['ensemble_size']
        else:
            ensemble_size = args.ensemble_size

        if args.image_size == None:
            image_size = model_config['image_size']
        else:
            image_size = args.image_size

        print(f"Using ensemble size: {ensemble_size} (variant: {variant})")

        model_output_dir = results_dir / model_name

        try:
            success, avg_loss = generate_predictions_for_model(
                model_path, variant, dataset, args.data_path, model_output_dir,
                ensemble_size, image_size, disable_ff_parser, disable_dynamic_encoding,
                calculate_loss=True, self_condition=args.self_condition
            )

            if success:
                loss_info = f" (avg loss: {avg_loss:.6f})" if avg_loss is not None else ""
                print(f"{model_name} completed{loss_info}")
            else:
                print(f"{model_name} failed")

        except Exception as e:
            print(f"{model_name} failed with error: {e}")

    print(f"\nPrediction generation completed!")
    print(f"Results saved in {results_dir}")


if __name__ == "__main__":
    main()
