import os
import argparse
from tqdm import tqdm
import jittor as jt
import numpy as np
from PIL import Image
from jittor import transform
from jittor import optim
import sys
sys.path.append("../")
sys.path.append("./")
from medsegdiff import Unet, GaussianDiffusion
from dataset_prepare import RefugeDataset, BraTsDataset, DDTIDataset
import wandb
import math

from medsegdiff import (
    initialize_jittor_for_cuda,
    ensure_float32_tensor,
    setup_model_float32,
    setup_diffusion_float32
)

# 使用CUDA和float32
initialize_jittor_for_cuda()

# CosineAnnealing学习率调度器
class CosineAnnealingScheduler:
    def __init__(self, optimizer, initial_lr, total_epochs, min_lr=1e-7, warmup_epochs=3, restart_epochs=None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.restart_epochs = restart_epochs  # 重启周期

        # 状态跟踪
        self.epoch = 0
        self.current_lr = initial_lr

        # 梯度爆炸处理
        self.explosion_threshold = 200.0
        self.explosion_count = 0
        self.max_explosions_per_epoch = 2

        print(f"CosineAnnealingScheduler initialized: initial_lr={initial_lr:.6e}, min_lr={min_lr:.6e}, total_epochs={total_epochs}")
        if restart_epochs:
            print(f"  - Restart every {restart_epochs} epochs")

    def step_epoch(self, loss=None):
        """每个epoch结束时调用"""
        self.epoch += 1
        self.explosion_count = 0  # 重置爆炸计数

        if self.epoch <= self.warmup_epochs:
            # Warmup阶段：线性增加学习率到初始值
            warmup_lr = self.initial_lr * (self.epoch / self.warmup_epochs)
            self.current_lr = warmup_lr
            self.optimizer.lr = self.current_lr
            print(f"Warmup epoch {self.epoch}/{self.warmup_epochs}: LR = {self.current_lr:.6e}")
            return

        # CosineAnnealing调度
        if self.restart_epochs and self.epoch > self.warmup_epochs:
            # 带重启的余弦退火
            effective_epoch = (self.epoch - self.warmup_epochs - 1) % self.restart_epochs
            effective_total = self.restart_epochs
        else:
            # 标准余弦退火
            effective_epoch = self.epoch - self.warmup_epochs - 1
            effective_total = self.total_epochs - self.warmup_epochs

        # 余弦退火公式
        cosine_factor = 0.5 * (1 + math.cos(math.pi * effective_epoch / effective_total))
        self.current_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor

        # 确保学习率不低于最小值
        self.current_lr = max(self.current_lr, self.min_lr)

        # 更新优化器学习率
        self.optimizer.lr = self.current_lr

        print(f"Epoch {self.epoch}: CosineAnnealing LR = {self.current_lr:.6e}")

    def handle_explosion(self, loss_value):
        """处理梯度爆炸，返回是否应该跳过这个batch"""
        if loss_value > self.explosion_threshold and self.explosion_count < self.max_explosions_per_epoch:
            self.explosion_count += 1
            # 在梯度爆炸时临时降低学习率
            if self.current_lr > self.min_lr * 10:
                old_lr = self.current_lr
                self.current_lr = max(self.current_lr * 0.95, self.min_lr)
                self.optimizer.lr = self.current_lr
                print(f"Explosion detected (loss={loss_value:.2f}): LR {old_lr:.6e} -> {self.current_lr:.6e}")
            return False  # 不跳过batch，继续训练
        elif loss_value > self.explosion_threshold:
            # 如果爆炸次数过多，跳过这个batch
            return True
        return False

    def get_lr(self):
        return self.current_lr

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'initial_lr': self.initial_lr,
            'current_lr': self.current_lr,
            'min_lr': self.min_lr,
            'total_epochs': self.total_epochs,
            'warmup_epochs': self.warmup_epochs,
            'restart_epochs': self.restart_epochs
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict.get('epoch', 0)
        self.initial_lr = state_dict['initial_lr']
        self.current_lr = state_dict.get('current_lr', self.initial_lr)
        self.min_lr = state_dict.get('min_lr', 1e-7)
        self.total_epochs = state_dict.get('total_epochs', self.total_epochs)
        self.warmup_epochs = state_dict.get('warmup_epochs', 3)
        self.restart_epochs = state_dict.get('restart_epochs', None)
        # 恢复优化器学习率
        self.optimizer.lr = self.current_lr

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='MedSegDiff Training Script')

    # 模型变体选择
    parser.add_argument('--model_variant', type=str, default='MedSegDiff-S',
                        choices=['MedSegDiff-S', 'MedSegDiff-B', 'MedSegDiff-L', 'MedSegDiff++'],
                        help='Model variant to train')

    # 消融实验选项
    parser.add_argument('--disable_ff_parser', action='store_true',
                        help='Disable FF-Parser module for ablation study')
    parser.add_argument('--disable_dynamic_encoding', action='store_true',
                        help='Disable Dynamic Conditional Encoding for ablation study')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (auto-set based on dataset if not specified)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--scale_lr', action='store_true',
                        help='Scale learning rate by batch size and gradient accumulation steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of gradient accumulation steps (default: 2)')
    parser.add_argument('--early_stopping_patience', type=int, default=30,
                        help='Early stopping patience (epochs without improvement, default: 10)')
    parser.add_argument('--min_delta', type=float, default=1e-6,
                        help='Minimum change in loss to qualify as improvement (default: 1e-6)')

    # 模型参数
    parser.add_argument('--self_condition', action='store_true',
                        help='Enable self conditioning in the model')
    parser.add_argument('--adam_beta1', type=float, default=0.9,
                        help='Adam optimizer beta1 parameter (default: 0.9)')
    parser.add_argument('--adam_beta2', type=float, default=0.999,
                        help='Adam optimizer beta2 parameter (default: 0.999)')
    parser.add_argument('--adam_weight_decay', type=float, default=1e-2,
                        help='Adam optimizer weight decay (default: 1e-2)')
    parser.add_argument('--adam_epsilon', type=float, default=1e-08,
                        help='Adam optimizer epsilon value (default: 1e-8)')

    # 数据参数
    parser.add_argument('--dataset', type=str, default='brats2021',
                        choices=['brats2021', 'refuge2', 'ddti'],
                        help='Dataset to use for training')
    parser.add_argument('--data_path', type=str, default='data/BraTs2021/Train',
                        help='Path to training data directory')
    parser.add_argument('--image_size', type=int, default=None,
                        help='Input image size (auto-set based on dataset if not specified)')
    parser.add_argument('--input_img_channels', type=int, default=None,
                        help='Number of input image channels (auto-set based on dataset)')
    parser.add_argument('--mask_channels', type=int, default=1,
                        help='Number of mask channels (default: 1)')

    # 扩散参数
    parser.add_argument('--timesteps', type=int, default=100,
                        help='Number of diffusion timesteps (default: 100)')
    parser.add_argument('--sampling_timesteps', type=int, default=100,
                        help='Number of sampling timesteps (default: 100, optimized)')

    # 输出和日志记录
    parser.add_argument('--output_dir', type=str, default='output/train',
                        help='Output directory for saving models and results')
    parser.add_argument('--logging_dir', type=str, default='logs',
                        help='Directory for logging files')
    parser.add_argument('--report_to', type=str, default='wandb', choices=['wandb', 'none'],
                        help='Logging platform (wandb or none)')

    # 保存和加载
    parser.add_argument('--save_every', type=int, default=1, help='save every n epochs (default: 1)')
    parser.add_argument('--load_model_from', default=None, help='path to pkl file to load from')
    parser.add_argument('--save_best_only', action='store_true', help='Only save the best model')

    # 批量训练选项
    parser.add_argument('--train_all_models', action='store_true',
                        help='Train all model variants')
    parser.add_argument('--train_variants', nargs='+', default=None,
                        choices=['MedSegDiff-S', 'MedSegDiff-B', 'MedSegDiff-L', 'MedSegDiff++'],
                        help='Specific model variants to train')
    parser.add_argument('--include_ablation', action='store_true',
                        help='Include ablation studies when training multiple variants')

    # 多数据集训练选项
    parser.add_argument('--train_all_datasets', action='store_true',
                        help='Train on all datasets (brats2021, refuge2, ddti)')
    parser.add_argument('--train_datasets', nargs='+', default=None,
                        choices=['brats2021', 'refuge2', 'ddti'],
                        help='Specific datasets to train on')

    # 性能优化
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable mixed precision training for speed up')

    # 学习率调度器参数
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'cosine_restart'],
                        help='Learning rate scheduler type (default: cosine)')
    parser.add_argument('--cosine_restart_epochs', type=int, default=None,
                        help='Restart period for cosine annealing with restarts (default: None)')

    return parser.parse_args()


def get_model_config(variant, dataset='brats2021'):
    """
    根据MedSegDiff论文内容，针对不同数据集进行适当的内存优化：
    论文中MedSegDiff-S、MedSegDiff-B、MedSegDiff-L和MedSegDiff++分别采用了下采样倍数为4倍、5倍、6倍和6倍的UNet结构；
    batch_size：MedSegDiff-B和MedSegDiff-S为32，MedSegDiff-L和MedSegDiff++为64；
    image_size：256×256；
    集成预测：25；
    初始学习率：0.0001；
    扩散推理步数：100；
    """
    if dataset in ['refuge2', 'ddti', 'brats2021']:
        configs = {
            'MedSegDiff-S': {
                'dim_mults': (1, 2, 4, 8),       # 4倍下采样
                'batch_size': 32,                # 批量大小
                'base_dim': 64,                  # 基础维度
                'image_size': 256,               # 图像尺寸
                'learning_rate_scale': 1.0,      # 标准学习率
                'description': f'Small variant'
            },
            'MedSegDiff-B': {
                'dim_mults': (1, 2, 4, 8, 16),   # 5倍下采样
                'batch_size': 32,
                'base_dim': 64,
                'image_size': 256,
                'learning_rate_scale': 1.0,       # 标准学习率
                'description': f'Base variant'
            },
            'MedSegDiff-L': {
                'dim_mults': (1, 2, 4, 8, 16, 32),  # 6倍下采样
                'batch_size': 16,                   # 降低批量大小
                'base_dim': 64,
                'image_size': 256,
                'learning_rate_scale': 1.0,
                'description': f'Large variant'
            },
            'MedSegDiff++': {
                'dim_mults': (1, 2, 4, 8, 16, 32),  # 6倍下采样
                'batch_size': 16,
                'base_dim': 64,
                'image_size': 256,
                'learning_rate_scale': 1.0,
                'description': f'Enhanced variant'
            }
        }

    config = configs[variant]
    return config


def load_data(args):
    """Load dataset based on configuration with proper data preprocessing"""
    print(f"Loading {args.dataset} dataset from: {args.data_path}")

    transform_list = [
        transform.Resize((args.image_size, args.image_size)),  # 尺寸调整
        transform.ToTensor(),                                  # 转换为张量
    ]
    transform_train = transform.Compose(transform_list)

    if args.dataset == 'brats2021':
        dataset = BraTsDataset(args.data_path, transform=transform_train, training=True)
        args.input_img_channels = 4  # t1, t1ce, t2, flair
        args.mask_channels = 1

    elif args.dataset == 'refuge2':
        dataset = RefugeDataset(args.data_path, transform=transform_train, training=True)
        args.input_img_channels = 3  # RGB
        args.mask_channels = 1

    elif args.dataset == 'ddti':
        dataset = DDTIDataset(args.data_path, transform=transform_train, training=True)
        args.input_img_channels = 3  # RGB
        args.mask_channels = 1

    else:
        raise NotImplementedError(f"The dataset {args.dataset} hasn't been implemented yet.")

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Input channels: {args.input_img_channels}, Mask channels: {args.mask_channels}")

    # 测试样本
    if len(dataset) > 0:       
        # REFUGE2/DDTI的额外验证
        if args.dataset in ['refuge2', 'ddti']:
            print(f"\nPerforming additional validation for {args.dataset}...")
            non_zero_masks = 0
            for i in range(len(dataset)):
                try:
                    sample = dataset[i]
                    if len(sample) >= 2:
                        mask = sample[1]
                        if not isinstance(mask, jt.Var):
                            if isinstance(mask, np.ndarray):
                                mask = jt.array(mask.astype(np.float32))
                            else:
                                mask = jt.array(mask)
                        mask = ensure_float32_tensor(mask)

                        if hasattr(mask, 'sum') and mask.sum() > 0:
                            non_zero_masks += 1
                except:
                    continue
            print(f"Non-zero masks found: {non_zero_masks}/{len(dataset)}")
            if non_zero_masks == 0:
                print("ERROR: All tested masks are zero! This will cause loss=0")

    else:
        print("Dataset appears to be empty!")
        print(f"   Dataset path: {args.data_path}")
        print(f"   Dataset database length: {len(dataset.database) if hasattr(dataset, 'database') else 'N/A'}")

    training_generator = dataset.set_attrs(
        batch_size=args.batch_size,
        shuffle=True)

    return training_generator


def create_model(args, model_config):
    """根据变体和消融设置创建模型"""

    # 确认是否使用Dynamic Conditional Encoding和FF-Parser
    skip_connect_condition_famps = not args.disable_dynamic_encoding
    dynamic_ff_parser_attn_map = not args.disable_ff_parser

    # 根据图像大小和下采样级别计算合适的补丁大小
    num_downsample = len(model_config['dim_mults']) - 1
    min_feature_size = args.image_size // (2 ** num_downsample)

    if args.model_variant == 'MedSegDiff++':
        # MedSegDiff++: Enhanced variant
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
        # 只在最后一层使用self-attention以避免未使用参数
        # 若self-attention在中间层冗余，其参数存在却未发挥作用会导致资源浪费
        # 仅在最后一层使用self-attention，可确保注意力机制仅作用于已提炼的高层特征
        full_self_attn = tuple([False] * (len(model_config['dim_mults']) - 1) + [True])
        mid_transformer_depth = 1
        attn_heads = 8
        attn_dim_head = 64

    elif args.model_variant == 'MedSegDiff-L':
        # MedSegDiff-L: Large variant
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
        # 只在最后一层使用self-attention
        full_self_attn = tuple([False] * (len(model_config['dim_mults']) - 1) + [True])
        mid_transformer_depth = 1
        attn_heads = 6
        attn_dim_head = 48

    elif args.model_variant == 'MedSegDiff-B':
        # MedSegDiff-B: Base variant
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
        # 只在最后一层使用self-attention
        full_self_attn = tuple([False] * (len(model_config['dim_mults']) - 1) + [True])
        mid_transformer_depth = 1
        attn_heads = 4
        attn_dim_head = 32
    else:
        # MedSegDiff-S: Small variant
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
        # 只在最后一层使用self-attention
        full_self_attn = tuple([False] * (len(model_config['dim_mults']) - 1) + [True])
        mid_transformer_depth = 1
        attn_heads = 2
        attn_dim_head = 16

    print(f"Model config: {args.model_variant}")
    print(f"  - Downsampling levels: {num_downsample}")
    print(f"  - Min feature size: {min_feature_size}")
    print(f"  - Patch size: {patch_size}")
    print(f"  - Attention heads: {attn_heads}")
    print(f"  - Attention dim: {attn_dim_head}")
    print(f"  - Transformer depth: {mid_transformer_depth}")
    print(f"  - FF-Parser enabled: {dynamic_ff_parser_attn_map}")
    print(f"  - Dynamic encoding enabled: {skip_connect_condition_famps}")

    model = Unet(
        dim=model_config['base_dim'],
        image_size=args.image_size,
        dim_mults=model_config['dim_mults'],
        full_self_attn=full_self_attn,
        attn_heads=attn_heads,
        attn_dim_head=attn_dim_head,
        mask_channels=args.mask_channels,
        input_img_channels=args.input_img_channels,
        self_condition=args.self_condition,
        skip_connect_condition_famps=skip_connect_condition_famps,  # 根据消融参数设置
        dynamic_ff_parser_attn_map=dynamic_ff_parser_attn_map,      # 根据消融参数设置
        mid_transformer_depth=mid_transformer_depth,
        conditioning_kwargs=conditioning_kwargs
    )

    model = setup_model_float32(model)
    if jt.flags.use_cuda:
        print("模型已配置为GPU训练模式")

    print(f"Advanced features configuration:")
    print(f"  - skip_connect_condition_famps: {'enabled' if skip_connect_condition_famps else 'disabled'}")
    print(f"  - dynamic_ff_parser_attn_map: {'enabled' if dynamic_ff_parser_attn_map else 'disabled'}")

    return model


def train_single_model(args, variant, disable_ff_parser=False, disable_dynamic_encoding=False):
    """Train a single model variant with automatic configuration"""

    # 清理之前的状态
    try:
        jt.clean()  # 清理Jittor缓存
        jt.gc()     # 垃圾回收
    except:
        pass

    args.model_variant = variant
    args.disable_ff_parser = disable_ff_parser
    args.disable_dynamic_encoding = disable_dynamic_encoding

    model_config = get_model_config(args.model_variant, args.dataset)

    # 调整数据路径
    if args.data_path == 'data/BraTs2021/Train': 
        if args.dataset == 'refuge2':
            args.data_path = 'data/REFUGE2'
        elif args.dataset == 'ddti':
            args.data_path = 'data/DDTI'
        elif args.dataset == 'brats2021':
            args.data_path = 'data/BraTs2021/Train'

    # 调整图像大小和批量大小
    if args.image_size is None:
        args.image_size = model_config['image_size']
    if args.batch_size is None:
        args.batch_size = model_config['batch_size']

    print(f"Training {args.model_variant}: {model_config['description']}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Base dimension: {model_config['base_dim']}")
    print(f"Dimension multipliers: {model_config['dim_mults']}")
    print(f"Sampling timesteps: {args.sampling_timesteps}")

    # 消融研究
    if args.disable_ff_parser:
        print("FF-Parser module DISABLED for ablation study")
    if args.disable_dynamic_encoding:
        print("Dynamic Conditional Encoding DISABLED for ablation study")

    variant_name = f"{args.model_variant}_{args.dataset}"
    if args.disable_ff_parser:
        variant_name += "_no_ff_parser"
    if args.disable_dynamic_encoding:
        variant_name += "_no_dynamic_encoding"

    checkpoint_dir = os.path.join(args.output_dir, variant_name, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 初始化wandb
    if args.report_to == "wandb":
        try:
            wandb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'wandb'))
            os.makedirs(wandb_dir, exist_ok=True)

            wandb.init(
                project="medsegdiff-jittor",
                name=f"{variant_name}",
                config=vars(args),
                dir=wandb_dir
            )
            print(f"WandB logging initialized, logs saved to: {wandb_dir}")
        except Exception as e:
            print(f"Warning: wandb initialization failed: {e}, continuing without logging")
            args.report_to = "none"

    ## 加载数据 ##
    data_loader = load_data(args)

    ## 定义模型 ##
    model = create_model(args, model_config)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 应用模型特定的学习率缩放
    base_scale = model_config.get('learning_rate_scale', 1.0)
    effective_lr = args.learning_rate * base_scale

    if args.scale_lr:
        # 进行学习率缩放
        scale_factor = args.batch_size * args.gradient_accumulation_steps
        scaled_lr = effective_lr * scale_factor
        max_lr = 0.0002  # 学习率上限
        effective_lr = min(scaled_lr, max_lr)
        print(f"Learning rate: {args.learning_rate} -> {effective_lr} (base_scale: {base_scale}, batch_scale: {args.batch_size}, grad_accum: {args.gradient_accumulation_steps}, total_scale: {scale_factor}, capped at {max_lr})")
    else:
        max_lr = 0.0003  # 非缩放情况下的最大学习率
        effective_lr = min(effective_lr, max_lr)
        print(f"Learning rate: {args.learning_rate} -> {effective_lr} (scale: {base_scale},  at capped{max_lr})")

    ## 初始化AdamW优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 使用CosineAnnealing学习率调度器
    restart_epochs = None
    if args.lr_scheduler == 'cosine_restart':
        restart_epochs = args.cosine_restart_epochs or max(args.epochs // 3, 10)

    scheduler = CosineAnnealingScheduler(
        optimizer=optimizer,
        initial_lr=effective_lr,
        total_epochs=args.epochs,
        min_lr=effective_lr * 1e-3,  # 最小学习率为初始值的1/1000
        warmup_epochs=min(3, args.epochs // 10),  # warmup轮数
        restart_epochs=restart_epochs
    )

    scheduler_type = "CosineAnnealingScheduler"
    if restart_epochs:
        scheduler_type += f" with restarts every {restart_epochs} epochs"
    print(f"Learning rate scheduler: {scheduler_type} (total_epochs={args.epochs}, min_lr={effective_lr * 1e-3:.2e})")

    ## 扩散模型 ##
    diffusion = GaussianDiffusion(
        model,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps
    )

    diffusion = setup_diffusion_float32(diffusion)

    # 如果指定，加载checkpoint
    start_epoch = 0
    best_loss = float('inf')
    if args.load_model_from is not None and os.path.exists(args.load_model_from):
        save_dict = jt.load(args.load_model_from)
        diffusion.model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])

        # 加载调度器状态
        if 'scheduler_state_dict' in save_dict:
            scheduler.load_state_dict(save_dict['scheduler_state_dict'])
            print("Loaded scheduler state from checkpoint")

        start_epoch = save_dict.get('epoch', 0) + 1
        best_loss = save_dict.get('best_loss', float('inf'))
        print(f'Loaded checkpoint from {args.load_model_from}, resuming from epoch {start_epoch}')

    ## 训练循环 ##
    print(f"\nStarting training for {args.epochs} epochs...")

    dataset_size = len(data_loader)
    total_batches_per_epoch = (dataset_size + args.batch_size - 1) // args.batch_size  # 向上取整
    total_steps = args.epochs * total_batches_per_epoch
    print(f"Dataset size: {dataset_size}")
    print(f"Total batches per epoch: {total_batches_per_epoch}")
    print(f"Total training steps: {total_steps}")

    # 早停策略
    patience_counter = 0
    best_loss_for_early_stop = float('inf') 

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        processed_samples = 0

        print(f'\nEpoch {epoch + 1}/{start_epoch + args.epochs}')

        # 创建进度条
        progress_bar = tqdm(
            enumerate(data_loader), 
            desc=f"Epoch {epoch + 1}/{start_epoch + args.epochs}",
            total=total_batches_per_epoch, 
            ncols=140, 
            leave=True,
            position=0,
            unit='batch'
        )

        for batch_idx, batch_data in progress_bar:

            if len(batch_data) == 2:
                img, mask = batch_data
            elif len(batch_data) == 3:
                img, mask, _ = batch_data
            else:
                print(f"Unexpected batch format: {len(batch_data)} elements")
                continue

            if not isinstance(img, jt.Var):
                if isinstance(img, np.ndarray):
                    img = jt.array(img.astype(np.float32))
                else:
                    img = jt.array(img)
            img = ensure_float32_tensor(img)

            if not isinstance(mask, jt.Var):
                if isinstance(mask, np.ndarray):
                    mask = jt.array(mask.astype(np.float32))
                else:
                    mask = jt.array(mask)
            mask = ensure_float32_tensor(mask)

            assert img.dtype == jt.float32, f"Image tensor must be float32, got {img.dtype}"
            assert mask.dtype == jt.float32, f"Mask tensor must be float32, got {mask.dtype}"

            if img.ndim == 5:
                # 从深度维度中取中间切片
                img = img[:, :, img.shape[2]//2, :, :]
                progress_bar.write(f"Reshaped image from 5D to 4D: {img.shape}")

            if mask.ndim == 5:
                # 从深度维度中取中间切片
                mask = mask[:, :, mask.shape[2]//2, :, :]
                progress_bar.write(f"Reshaped mask from 5D to 4D: {mask.shape}")
            elif mask.ndim == 4 and mask.shape[1] > 1:
                # 如果掩码有多个通道，则取第一个
                mask = mask[:, 0:1, :, :]
                progress_bar.write(f"Reduced mask channels: {mask.shape}")

            # 如果掩码全为零则跳过该批处理（将导致损失为零）
            if mask.sum() == 0:
                progress_bar.write(f"WARNING: Batch {batch_idx} has zero mask, skipping...")
                continue

            # 初始化梯度
            optimizer.zero_grad()

            try:
                mask_input = ensure_float32_tensor(mask)
                img_input = ensure_float32_tensor(img)

                if hasattr(mask_input, 'dtype') and hasattr(img_input, 'dtype'):
                    if mask_input.dtype != jt.float32 or img_input.dtype != jt.float32:
                        mask_input = mask_input.float32()
                        img_input = img_input.float32()

                # 计算扩散损失
                loss = diffusion(mask_input, img_input)

                if hasattr(loss, 'dtype') and loss.dtype != jt.float32:
                    loss = loss.float32()

                if jt.isnan(loss).any() or jt.isinf(loss).any():
                    progress_bar.write(f"WARNING: Invalid loss in batch {batch_idx}: {loss.item()}")
                    continue

                # 使用自定义调度器处理梯度爆炸问题
                should_skip = scheduler.handle_explosion(loss.item())
                if should_skip:
                    progress_bar.write(f"Skipping batch due to repeated explosions")
                    continue

                # 反向传播
                optimizer.backward(loss)

                # 梯度裁剪
                total_norm = 0.0
                for param in model.parameters():
                    try:
                        grad = param.opt_grad(optimizer)
                        if grad is not None:
                            param_norm = grad.norm()
                            total_norm += param_norm.item() ** 2
                    except:
                        continue

                total_norm = total_norm ** 0.5

                # 在梯度范数过大时进行裁剪
                if total_norm > 5.0:  
                    clip_coef = 5.0 / (total_norm + 1e-6)
                    for param in model.parameters():
                        try:
                            grad = param.opt_grad(optimizer)
                            if grad is not None:
                                grad.data = grad.data * clip_coef
                        except:
                            continue

                # 更新优化器
                optimizer.step()

                # 计算最终损失用于记录
                final_loss = loss.item()

            except Exception as e:
                progress_bar.write(f"ERROR in training step for batch {batch_idx}: {e}")
                continue

            running_loss += final_loss
            num_batches += 1
            processed_samples += img.shape[0]  # 添加实际批量大小

            # 显示每个step的损失、平均损失、已处理的样本数、学习率
            progress_bar.set_postfix({
                'loss': f'{final_loss:.6f}',
                'avg_loss': f'{running_loss/(num_batches):.6f}',
                'sample': f'{num_batches*args.batch_size}/{total_batches_per_epoch*args.batch_size}',
                'lr': f'{scheduler.get_lr():.2e}'
            })

            # 记录到wandb
            if args.report_to == "wandb":
                try:
                    wandb.log({
                        'batch_loss': final_loss,
                        'mask_sum': mask.sum().item(),
                        'image_mean': img.mean().item(),
                        'epoch': epoch + 1,
                        'batch': batch_idx
                    })
                except:
                    pass

        progress_bar.refresh()
        progress_bar.close()

        # 计算每一轮的损失
        epoch_loss = running_loss / max(num_batches, 1)

        # 获取当前学习率
        current_lr = scheduler.get_lr()

        print(f'Epoch {epoch + 1}/{start_epoch + args.epochs} completed - Average Loss: {epoch_loss:.4f} (Batches: {num_batches}) - LR: {current_lr:.2e}')

        # 使用epoch损失更新学习率调度器
        scheduler.step_epoch(epoch_loss)

        # 记录每一轮的指标
        if args.report_to == "wandb":
            try:
                wandb.log({
                    'epoch_loss': epoch_loss,
                    'epoch': epoch + 1,
                    'learning_rate': current_lr
                })
            except:
                pass

        # 早停策略
        improvement_threshold = args.min_delta
        if epoch < 20: 
            improvement_threshold = args.min_delta * 0.1

        if epoch_loss < best_loss_for_early_stop - improvement_threshold:
            best_loss_for_early_stop = epoch_loss
            patience_counter = 0
            print(f"New best loss: {best_loss_for_early_stop:.6f}")
        else:
            patience_counter += 1

        # 早停策略
        if epoch >= 40 and patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
            print(f"Best loss: {best_loss_for_early_stop:.6f}")
            break
        elif patience_counter >= args.early_stopping_patience:
            print(f"Would trigger early stopping, but still in initial training phase (epoch {epoch+1})")
            print(f"Patience counter: {patience_counter}/{args.early_stopping_patience}")

        # 保存checkpoint
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss

        if (epoch + 1) % args.save_every == 0 or is_best:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': diffusion.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'best_loss': best_loss,
                'args': vars(args)
            }

            if args.save_best_only and is_best:
                save_path = os.path.join(checkpoint_dir, f'{variant_name}_best.pkl')
                jt.save(save_dict, save_path)
                print(f'Best model saved: {save_path}')
            elif not args.save_best_only:
                save_path = os.path.join(checkpoint_dir, f'{variant_name}_epoch_{epoch+1}_loss_{epoch_loss:.4f}.pkl')
                jt.save(save_dict, save_path)
                print(f'Checkpoint saved: {save_path}')

    print(f"\nTraining completed! Best loss: {best_loss:.4f}")

    # 关闭wandb
    if args.report_to == "wandb":
        try:
            wandb.finish()
        except:
            pass

    # 清理资源
    try:
        del model
        del diffusion
        del optimizer
        jt.clean()
        jt.gc()
    except:
        pass

    return best_loss


def main():
    """Main function to handle single or batch training"""
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.train_all_datasets:
        datasets = ['brats2021', 'refuge2', 'ddti']
    elif args.train_datasets:
        datasets = args.train_datasets
    else:
        datasets = [args.dataset]

    if args.train_all_models:
        variants = ['MedSegDiff-S', 'MedSegDiff-B', 'MedSegDiff-L', 'MedSegDiff++']
    elif args.train_variants:
        variants = args.train_variants
    else:
        variants = [args.model_variant]

    if len(datasets) > 1 or len(variants) > 1 or args.train_all_models or args.include_ablation:
        print(f"Starting comprehensive training:")
        print(f"  Datasets: {datasets}")
        print(f"  Variants: {variants}")
        print(f"  Include ablation: {args.include_ablation}")

        results = {}

        for dataset in datasets:
            print(f"\n{'='*80}")
            print(f"TRAINING ON {dataset.upper()} DATASET")
            print(f"{'='*80}")

            original_dataset = args.dataset
            args.dataset = dataset

            dataset_results = {}

            for variant in variants:
                print(f"\n{'='*60}")
                print(f"Training {variant} on {dataset}")
                print(f"{'='*60}")

                # 训练基础模型
                try:
                    best_loss = train_single_model(args, variant, args.disable_ff_parser, args.disable_dynamic_encoding)
                    dataset_results[f"{variant}_{dataset}"] = best_loss
                    print(f"{variant} on {dataset} completed with best loss: {best_loss:.4f}")
                except Exception as e:
                    print(f"{variant} on {dataset} failed: {e}")
                    dataset_results[f"{variant}_{dataset}"] = None

                # 进行消融研究
                if args.include_ablation:
                    ablation_configs = [
                        (True, False, "no_ff_parser"),
                        (False, True, "no_dynamic_encoding"),
                        (True, True, "no_ff_parser_no_dynamic_encoding")
                    ]

                    for disable_ff, disable_dyn, suffix in ablation_configs:
                        print(f"\n{'-'*40}")
                        print(f"Training {variant} - {suffix} on {dataset}")
                        print(f"{'-'*40}")

                        try:
                            best_loss = train_single_model(args, variant, disable_ff, disable_dyn)
                            dataset_results[f"{variant}_{suffix}_{dataset}"] = best_loss
                            print(f"{variant}_{suffix} on {dataset} completed with best loss: {best_loss:.4f}")
                        except Exception as e:
                            print(f"{variant}_{suffix} on {dataset} failed: {e}")
                            dataset_results[f"{variant}_{suffix}_{dataset}"] = None

            results.update(dataset_results)

            # 恢复原始数据集设置
            args.dataset = original_dataset

        print(f"\n{'='*80}")
        print("COMPREHENSIVE TRAINING SUMMARY")
        print(f"{'='*80}")

        for dataset in datasets:
            print(f"\n{dataset.upper()} DATASET:")
            print("-" * 50)
            dataset_results = {k: v for k, v in results.items() if k.endswith(f"_{dataset}")}
            for model_name, loss in dataset_results.items():
                status = f"{loss:.4f}" if loss is not None else "FAILED"
                clean_name = model_name.replace(f"_{dataset}", "")
                print(f"  {clean_name:35} | {status}")

    else:
        # 在单个数据集上训练单个模型
        print(f"Training single model: {args.model_variant} on {args.dataset}")
        try:
            best_loss = train_single_model(args, args.model_variant, args.disable_ff_parser, args.disable_dynamic_encoding)
            print(f"Training completed with best loss: {best_loss:.4f}")
        except Exception as e:
            print(f"Training failed: {e}")


if __name__ == '__main__':
    main()
