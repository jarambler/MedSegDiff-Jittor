import jittor as jt
from jittor import nn
from .utils import exists, default, LayerNorm

class Block(nn.Module):
    """
    基本块，包含一个卷积层、一个层归一化层和一个激活函数
    param dim: 输入张量的通道数
    param dim_out: 输出张量的通道数, 默认为dim
    param groups: 分组卷积的组数, 默认为8
    """
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        # 卷积层，核大小为3*3，步长为1，填充为1，保持特征图尺寸不变
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        # 分组归一化层，将dim_out通道分成groups组分别归一化，加速训练并提升稳定性
        self.norm = nn.GroupNorm(groups, dim_out)
        # Sigmoid加权线性单元激活函数
        self.act = nn.SiLU()

    # 重写execute方法, 前向传播
    def execute(self, x, scale_shift=None):
        # 卷积层
        x = self.proj(x)
        # 分组归一化层
        x = self.norm(x)
        # 通道注意力缩放和偏移
        if exists(scale_shift):
            scale, shift = scale_shift
            # scale + 1避免失去原始特征
            x = x * (scale + 1) + shift
        # SiLU激活函数
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """
    ResNet块，包含两个基本块和一个可选的下采样或上采样操作
    param dim: 输入张量的通道数
    param dim_out: 输出张量的通道数, 默认为dim
    param time_emb_dim: 时间嵌入的维度, 默认为None
    param groups: 分组卷积的组数, 默认为8
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        # 时间嵌入模块
        self.mlp = nn.Sequential(
            nn.SiLU(),
            # 将时间步嵌入映射到2*dim_out，分别用于scale和shift
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        # 第一个基本块，负责特征维度的转换，过渡层
        self.block1 = Block(dim, dim_out, groups=groups)
        # 第二个基本块，进一步处理特征，主处理层
        self.block2 = Block(dim_out, dim_out, groups=groups)
        # 残差连接的卷积层，如果输入输出通道数不一致，用1×1卷积投影，否则使用恒等映射
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def execute(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            # 提取时间嵌入的缩放和偏移
            time_emb = self.mlp(time_emb)
            # 重塑为[time_emb[0], time_emb[1], 1, 1]，便于广播
            time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)
            # 分为scale和shift两部分
            scale_shift = time_emb.chunk(2, dim=1)

        # 时间嵌入于第一个卷积块中
        h = self.block1(x, scale_shift=scale_shift)
        # 第二个卷积块
        h = self.block2(h)
        # 残差连接
        return h + self.res_conv(x)
        
def FeedForward(dim, mult=4):
    """
    前馈网络模块，包含两个1x1卷积层和GELU激活函数
    param dim: 输入张量的通道数
    param mult: 第一个卷积层的输出通道数相较于输入的倍数, 默认为4
    """
    # 计算第二个卷积层的输入通道数
    inner_dim = int(dim * mult)
    return nn.Sequential(
        # 层归一化
        LayerNorm(dim),
        nn.Conv2d(dim, inner_dim, 1),
        # GELU激活函数，GELU(x) = x·P(X ≤ x)​​
        nn.GELU(),
        nn.Conv2d(inner_dim, dim, 1),
    )