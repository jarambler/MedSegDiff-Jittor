import jittor as jt
from jittor import nn
from .utils import exists, default, LayerNorm
from .attention import Transformer

class ViT(nn.Module):
    """
    Vision Transformer模块，用于图像分类和特征提取
    param dim: 输入张量的通道数
    param image_size: 输入图像的尺寸
    param patch_size: 分块大小
    param channels: 输入图像的通道数, 默认为3
    param channels_out: 输出图像的通道数, 默认为None
    param heads: 注意力头数, 默认为4
    param dim_head: 每个注意力头的维度, 默认为32
    param depth: Transformer的层数, 默认为4
    """
    def __init__(
            self,
            dim,
            *,
            image_size,
            patch_size,
            channels=3,
            channels_out=None,
            heads=4,
            dim_head=32,
            depth=4                  
    ):
        super().__init__()
        # 图像尺寸需为正数
        assert exists(image_size)
        # 图像尺寸需能整除patch
        assert(image_size % patch_size) == 0
        # 每个维度上的patch数量
        num_patches_height_width = image_size // patch_size

        # 位置嵌入
        self.pos_emb = jt.zeros((dim, num_patches_height_width, num_patches_height_width))

        # 若未指定输出通道数，则默认与输入通道数相同
        channels_out = default(channels_out, channels)
        
        # 计算patch的维度
        patch_dim = channels * (patch_size ** 2)
        otput_patch_dim = channels_out * (patch_size ** 2)

        class PatchEmbedding(nn.Module):
            """
            重排模块，将图像切分为patches，并重塑为[B, C, P1, P2, H, W] -> [B, C*P1*P2, H/P1, W/P2]
            """
            def __init__(self, dim, patch_size, channels):
                super().__init__()
                self.patch_size = patch_size                

            def execute(self, x):
                b, c, h, w = x.shape
                p1 = p2 = self.patch_size
                x = x.view(b, c, h//p1, p1, w//p2, p2)
                x = x.permute(0, 1, 3, 5, 2, 4)
                x = x.contiguous().view(b, c*p1*p2, h//p1, w//p2)
                return x
            
        class PatchReconstrution(nn.Module):
            """
            重排模块，将patches重塑为图像，并重塑为[B, C*P1*P2, H/P1, W/P2] -> [B, C, P1, P2, H, W] -> [B, C, H, W]
            """
            def __init__(self, patch_size):
                super().__init__()
                self.patch_size = patch_size

            def execute(self, x):
                b, cpatch, h, w = x.shape
                p1 = p2 = self.patch_size
                c = cpatch // (p1 * p2)
                x = x.view(b, c, p1, p2, h, w)
                x = x.permute(0, 1, 4, 2, 5, 3)
                x = x.contiguous().view(b, c, h*p1, w*p2)
                return x
        
        # 图像切分为patches，并重塑为[B, C, P1, P2, H, W] -> [B, C*P1*P2, H/P1, W/P2]
        self.to_tokens = nn.Sequential(
            PatchEmbedding(dim, patch_size, channels),
            nn.Conv2d(patch_dim, dim, 1),
            LayerNorm(dim)
        )

        # Transformer模块，用于处理patches
        self.transformer = Transformer(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            depth=depth
        )

        # 将patches重塑为图像，并重塑为[B, C*P1*P2, H/P1, W/P2] -> [B, C, P1, P2, H, W] -> [B, C, H, W]
        self.to_patches = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, otput_patch_dim, 1),
            PatchReconstrution(patch_size)
        )

        nn.init.constant_(self.to_patches[1].weight, 0)
        nn.init.constant_(self.to_patches[1].bias, 0)

    # 前向传播
    def execute(self, x):
        x = self.to_tokens(x)
        x = x + self.pos_emb
        x = self.transformer(x)
        x = self.to_patches(x)
        return x