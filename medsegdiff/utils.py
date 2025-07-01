import math
import jittor as jt
from jittor import nn

# 辅助工具
# 判断变量是否存在
def exists(x):
    return x is not None

# 为空值赋默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d # d可调用则调用后返回，否则直接返回d

# 恒等函数
def identity(t, *args, **kwargs):
    return t

# 图像归一化函数
# [0,1]图像归一化到[-1,1]
def normalize_to_neg_one_to_one(img):
    # Ensure input is float32 and output is float32
    if hasattr(img, 'float32'):
        img = img.float32()
    result = img * 2.0 - 1.0
    if hasattr(result, 'float32'):
        result = result.float32()
    return result

# [-1,1]数据还原到[0,1]
def unnormalize_to_zero_to_one(t):
    # Ensure input is float32 and output is float32
    if hasattr(t, 'float32'):
        t = t.float32()
    result = (t + 1.0) * 0.5
    if hasattr(result, 'float32'):
        result = result.float32()
    return result

# 扩散辅助函数
# 从时间调度向量中提取batch个样本在t时间步对应的超参数值
def extract(a, t, x_shape):
    b, *_ = t.shape  # 获取t的batch size
    out = a.gather(-1, t)  # 从张量a最后一个维度按t中的时间索引提取值
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # 将out重塑为与x_shape匹配的形状

# 构造线性beta(噪声强度)调度表，转为float32
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    result = jt.linspace(beta_start, beta_end, timesteps)
    return result.float32() if hasattr(result, 'float32') else result

# 平滑非线性加噪调度表，转为float32
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = jt.linspace(0, timesteps, steps)
    if hasattr(x, 'float32'):
        x = x.float32()

    # alphas_cumprod：累计乘积
    # 余弦型schedule曲线，从1平滑地减小到接近0，相比linear，前期下降更慢，后期下降更快
    alphas_cumprod = jt.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    # 保证alpha_0 = 1，表示无噪声的情况，即原图
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # 反推加噪过程每一步的beta
    # beta = 1 - alpha_t / alpha_{t-1}
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    # 限制beta范围, beta<0 -> 0, beta>0.999 -> 0.999
    result = jt.clamp(betas, 0, 0.999)
    return result.float32() if hasattr(result, 'float32') else result

class Residual(nn.Module):
    """
    残差模块，将输入x与函数fn(x)的输出相加
    param fn: 一个函数，接受输入x并返回一个张量
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    # 重写execute方法, 将原始输入与经fn(x)的输出相加
    def execute(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
class LayerNorm(nn.Module):
    """
    层归一化模块，对输入张量进行层归一化
    param dim: 输入张量的维度
    """
    def __init__(self, dim, bias=False):
        super().__init__()
        # 强制使用float32
        self.gamma = jt.ones((1, dim, 1, 1), dtype=jt.float32)  # 缩放系数gamma
        self.beta = jt.zeros((1, dim, 1, 1), dtype=jt.float32) if bias else None  # 偏置系数beta
        
    # 重写execute方法, 对输入张量进行层归一化
    def execute(self, x):
        # 防止除0, 精度<float32 -> 1e-3
        eps = 1e-5 if x.dtype == jt.float32 else 1e-3
        # 计算每个位置上所有channel的方差和均值
        # unbiased=False使用总体方差(分母为n)，=True使用样本方差(分母为n-1)
        # keepdims=True保留维度，除dim外其余维度形状与x一致(dim维度变为1)
        var = jt.var(x, dim=1, unbiased=False, keepdims=True)  # 方差
        mean = jt.mean(x, dim=1, keepdims=True)  # 均值
        # 归一化公式
        # LayerNorm(x) = ((x - mean) / sqrt(var + eps)) * gamma + beta
        return (x - mean) * (var + eps).rsqrt() * self.gamma + default(self.beta, 0)
        
class SinusoidalPosEmb(nn.Module):
    """
    位置编码模块，将时间步t映射到一个可学习的向量
    param dim: 嵌入维度
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    # 重写execute方法, 将时间步t映射到一个可学习的向量
    def execute(self, x):
        # 位置编码公式
        # PE(t, 2i) = sin(t / 10000^(2i/dim))
        # PE(t, 2i+1) = cos(t / 10000^(2i/dim))
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # Ensure all tensors are float32
        emb = jt.exp(jt.arange(half_dim, dtype=jt.float32) * -emb)
        # Ensure input x is float32
        x = x.float32()
        # out_{i,j} = t_i * freq_j
        # [batch, 1] * [1, half_dim] = [batch, half_dim]
        emb = x[:, None] * emb[None, :]
        # 拼接sin和cos, [batch, half_dim] -> [batch, dim]
        emb = jt.concat((emb.sin(), emb.cos()), dim=-1)
        # Ensure output is float32
        return emb.float32()
    
def Upsample(dim, dim_out=None):
    """
    上采样模块，将输入张量的尺寸扩大两倍
    param dim: 输入张量的通道数
    param dim_out: 输出张量的通道数，默认为dim
    """
    return nn.Sequential(
        # 使用最近邻差值，将输入的高和宽放大两倍
        nn.Upsample(scale_factor=2, mode='nearest'),
        # 3*3卷积层，将通道数从dim调整到dim_out
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    """
    下采样模块，将输入张量的尺寸减小两倍
    param dim: 输入张量的通道数
    param dim_out: 输出张量的通道数, 默认为dim
    """
    class Rearrange(nn.Module):
        """
        重排模块，通道数乘4，高和宽除以2
        b c h w -> b c (h 2) (w 2) -> b (c 2 2) h w
        """
        def execute(self, x):
            b, c, h, w = x.shape
            x = x.view(b, c, h//2, 2, w//2, 2)  # 拆分特征图
            x = x.permute(0, 1, 3, 5, 2, 4)  # 调整维度顺序
            # 先确保张量内存布局连续，然后重塑张量形状
            x = x.contiguous().view(b, c*4, h//2, w//2)  
            return x
        
    return nn.Sequential(
        # 通道数乘4，高和宽除以2
        Rearrange(),
        # 1*1卷积层，将通道数从dim*4调整到dim_out
        nn.Conv2d(dim*4, default(dim_out, dim), 1)
    )