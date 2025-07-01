import jittor as jt
from jittor import nn
from .utils import LayerNorm, Residual
from .blocks import FeedForward

class LinearAttention(nn.Module):
    """
    线性注意力模块，用于计算注意力权重
    param dim: 输入张量的通道数
    param heads: 注意力头数, 默认为4
    param dim_head: 每个注意力头的维度, 默认为32
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        # 用于缩放q，避免数值爆炸
        self.scale = dim_head ** -0.5
        self.heads = heads
        # 多头并行
        hidden_dim = dim_head * heads

        # 层归一化
        self.prenorm = LayerNorm(dim)
        # 通道注意力缩放和偏移，3个卷积层，每个卷积层的输出通道数为hidden_dim
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)        
        
        # 将注意力输出重新投影回原始维度
        self.to_out = nn.Sequential(
            # 通道注意力输出，1个卷积层，输出通道数为dim
            nn.Conv2d(hidden_dim, dim, 1),
            # 层归一化
            LayerNorm(dim)
        )

    # 重写execute方法, 前向传播
    def execute(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)

        q, k, v = [t.view(b, self.heads, -1, h * w) for t in qkv]

        q = nn.softmax(q, dim=-2)
        k = nn.softmax(k, dim=-1)

        q = q * self.scale

        context = jt.matmul(k.transpose(-2, -1), v)
        out = jt.matmul(q, context)
        out = out.reshape(b, -1, h, w)

        return self.to_out(out)
        
class Attention(nn.Module):
    """
    自注意力模块，用于计算注意力权重
    param dim: 输入张量的通道数
    param heads: 注意力头数, 默认为4
    param dim_head: 每个注意力头的维度, 默认为32
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        # 用于缩放q，避免数值爆炸
        self.scale = dim_head ** -0.5
        self.heads = heads
        # 多头并行
        hidden_dim = dim_head * heads

        # 层归一化
        self.prenorm = LayerNorm(dim)
        # 通道注意力缩放和偏移，3个卷积层，每个卷积层的输出通道数为hidden_dim
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        # 将注意力输出重新投影回原始维度
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    # 重写execute方法, 前向传播
    def execute(self, x):
        b, c, h, w = x.shape

        x = self.prenorm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = [t.view(b, self.heads, -1, h * w) for t in qkv]

        q = q * self.scale

        sim = jt.matmul(q, k.transpose(-2, -1))
        attn = nn.softmax(sim, dim=-1)
        out = jt.matmul(attn, v)

        out = out.view(b, -1, h, w)
        return self.to_out(out)
    
class Transformer(nn.Module):
    """
    Transformer模块，包含自注意力和前馈网络
    param dim: 输入张量的通道数      
    param heads: 注意力头数, 默认为4
    param dim_head: 每个注意力头的维度, 默认为32 
    param depth: Transformer的层数
    """
    def __init__(self, dim, heads=4, dim_head=32, depth=1):
        super().__init__()
        # 多层Transformer
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(Attention(dim, heads=heads, dim_head=dim_head)),
                Residual(FeedForward(dim))
            ]))        
        
    # 重写execute方法, 前向传播
    def execute(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x