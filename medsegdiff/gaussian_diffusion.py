import math
from collections import namedtuple
from functools import partial
import random
import jittor as jt
from jittor import nn
from tqdm.auto import tqdm
from .utils import (
    exists, default, identity, normalize_to_neg_one_to_one, 
    unnormalize_to_zero_to_one, extract, linear_beta_schedule, 
    cosine_beta_schedule
)
from .unet import Unet

# pred_noise:模型预测的噪声  pred_x_start:模型预测的xt起始图像
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class GaussianDiffusion(nn.Module):
    """
    扩散模型
    param model: U-Net模型
    param timesteps: 扩散步数, 默认为100
    param sampling_timesteps: 采样步数, 默认为None
    param objective: 损失函数目标, 默认为'pred_noise'
    param beta_schedule: 噪声调度, 默认为'cosine'
    param ddim_sampling_eta: DDIM采样时的eta, 默认为1.0
    """
    def __init__(
            self,
            model,
            *,
            timesteps=100,
            sampling_timesteps=None,
            objective='pred_noise',
            beta_schedule='cosine',
            ddim_sampling_eta=1.
    ):
        super().__init__()

        if hasattr(model, 'module'):
            # 如果模型被包装(例如DataParallel)，提取底层模型
            self.model = model.module
        else:
            # 直接使用模型
            self.model = model        

        self.input_img_channels = self.model.input_img_channels
        self.mask_channels = self.model.mask_channels
        self.self_condition = self.model.self_condition
        self.image_size  = self.model.image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, \
            'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # 转换为float32
        alphas = (1. - betas).float32()
        alphas_cumprod = jt.cumprod(alphas, dim=0).float32()
        alphas_cumprod_prev = jt.concat([jt.ones(1, dtype=jt.float32), alphas_cumprod[:-1]], dim=0).float32()
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps

        self.ddim_sampling_eta = ddim_sampling_eta

        # 确保所有缓冲区都是float32类型
        register_buffer = lambda name, val: setattr(self, name, val.float32() if hasattr(val, 'float32') else val.float())

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 用于反向传播时计算损失函数
        register_buffer('sqrt_alphas_cumprod', jt.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', jt.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', jt.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', jt.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', jt.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # 用于采样时计算xt-1
        register_buffer('posterior_log_variance_clipped', jt.log(jt.clamp(posterior_variance, min_v=1e-20)))
        register_buffer('posterior_mean_coef1', betas * jt.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * jt.sqrt(alphas) / (1. - alphas_cumprod))

    # 预测xt-1
    def predict_start_from_noise(self, x_t, t, noise):
        return(
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # 预测噪声
    def predict_noise_from_start(self, x_t, t, x0):
        return(
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / 
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    # 预测v
    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )
    
    # 预测xt-1
    def predict_start_from_v(self, x_t, t, v):
        return(
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    # 计算q(xt-1 | xt, x0)
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # 计算模型预测
    def model_predictions(self, x, t, c, x_self_cond=None, clip_x_start=False):
        model_output = self.model(x, t, c, x_self_cond)
        maybe_clip = partial(jt.clamp, min_v=-1., max_v=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise =model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        
        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        
        return ModelPrediction(pred_noise, x_start)
    
    # 计算模型均值方差
    def p_mean_variance(self, x, t, c, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, c, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start = jt.clamp(x_start, -1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    # 采样xt-1
    def p_sample(self, x, t, c, x_self_cond=None, clip_denoised=True):
        b, *_, = x.shape
        batched_times = jt.full((x.shape[0],), t, dtype=jt.int64)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, c=c, x_self_cond=x_self_cond, clip_denoised=clip_denoised)
        noise = jt.randn_like(x).float32() if t > 0 else jt.zeros_like(x).float32()
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    # 采样过程
    def p_sample_loop(self, shape, cond):
        batch = shape[0]

        img = jt.randn(shape, dtype=jt.float32)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, cond, self_cond)

        img = unnormalize_to_zero_to_one(img)
        return img
    
    # DDIM采样
    def ddim_sample(self, shape, cond_img, clip_denoised=True):
        batch, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = jt.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = jt.randn(shape, dtype=jt.float32)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = jt.full((batch,), time, dtype=jt.int64)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond_img, self_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = jt.randn_like(img).float32()

            img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

        img = unnormalize_to_zero_to_one(img)
        return img
    
    # 采样
    def sample(self, cond_img):
        batch_size = cond_img.shape[0]

        image_size, mask_channels = self.image_size, self.mask_channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, mask_channels, image_size, image_size), cond_img)
    
    # 计算q(xt | x0)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: jt.randn_like(x_start).float32())

        return(
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    # 计算损失函数
    def p_losses(self, x_start, t, cond, noise=None):
        b, c, h, w = x_start.shape

        # 确保所有输入都是float32类型
        x_start = x_start.float32()
        cond = cond.float32()

        noise = default(noise, lambda: jt.randn_like(x_start).float32())
        if hasattr(noise, 'float32'):
            noise = noise.float32()

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_self_cond = None
        if self.self_condition and random.random() < 0.5:
            x_self_cond = self.model_predictions(x, t, cond).pred_x_start

        model_out = self.model(x, t, cond, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        loss = nn.mse_loss(model_out, target)
        return loss
    
    # 重写execute方法, 前向传播
    def execute(self, img, cond_img, *args, **kwargs):
        # 处理不同的输入维度
        if img.ndim == 3:
            img = img.unsqueeze(1)
        elif img.ndim == 5:
            # 对于BraTS数据: [batch, channels, depth, height, width] -> [batch, channels, height, width]
            # 取中间切片或沿深度取平均
            img = img[:, :, img.shape[2]//2, :, :]  # 取中间切片

        if cond_img.ndim == 3:
            cond_img = cond_img.unsqueeze(1)
        elif cond_img.ndim == 5:
            # 对于BraTS数据: [batch, channels, depth, height, width] -> [batch, channels, height, width]
            # 取中间切片或沿深度取平均
            cond_img = cond_img[:, :, cond_img.shape[2]//2, :, :]  # 取中间切片

        # 提取形状信息
        img_shape = img.shape
        cond_img_shape = cond_img.shape

        if len(img_shape) != 4 or len(cond_img_shape) != 4:
            raise ValueError(f"Expected 4D tensors after preprocessing, got img: {img_shape}, cond_img: {cond_img_shape}")

        b, c, h, w = img_shape
        img_size, img_channels, mask_channels = self.image_size, self.input_img_channels, self.mask_channels

        # 维度检查，确保输入图像尺寸与模型期望的尺寸一致
        if h != img_size or w != img_size:
            print(f"Warning: Image size mismatch. Expected {img_size}x{img_size}, got {h}x{w}. Resizing...")
            
            img = jt.nn.interpolate(img, size=(img_size, img_size), mode='bilinear', align_corners=False)
            cond_img = jt.nn.interpolate(cond_img, size=(img_size, img_size), mode='bilinear', align_corners=False)
            h, w = img_size, img_size

        assert cond_img_shape[1] == img_channels, f'your input medical must have {img_channels} channels, got {cond_img_shape[1]}'
        assert img_shape[1] == mask_channels, f'the segmented image must have {mask_channels} channels, got {img_shape[1]}'

        times = jt.randint(0, self.num_timesteps, (b,), dtype=jt.int64)

        # 确保所有张量都是float32类型
        img = img.float32()
        cond_img = cond_img.float32()

        img = normalize_to_neg_one_to_one(img)

        return self.p_losses(img, times, cond_img, *args, **kwargs)