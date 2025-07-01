import numpy as np
import jittor as jt
from jittor import nn
from .utils import LayerNorm
from .blocks import ResnetBlock
from .vit import ViT

def fft2(x):
    """使用numpy的2D FFT，返回单独的实部和虚部"""
    x_np = x.detach().numpy()
    x_fft = np.fft.fft2(x_np, axes=(-2, -1))
    # 返回一个复数对象，包含单独的实部和虚部张量
    return ComplexTensor(jt.array(x_fft.real.astype(np.float32)),
                        jt.array(x_fft.imag.astype(np.float32)))

def ifft2(x_complex):
    """使用numpy的2D IFFT，返回单独的实部和虚部"""
    if isinstance(x_complex, ComplexTensor):
        real_np = x_complex.real.detach().numpy()
        imag_np = x_complex.imag.detach().numpy()
        complex_np = real_np + 1j * imag_np
    else:
        # x_complex是实数张量的情况
        complex_np = x_complex.detach().numpy()

    x_ifft = np.fft.ifft2(complex_np, axes=(-2, -1))
    return ComplexTensor(jt.array(x_ifft.real.astype(np.float32)),
                        jt.array(x_ifft.imag.astype(np.float32)))

def view_as_real(x_complex):
    """将复数张量转换为实数张量，最后一个维度用于实部/虚部"""
    if isinstance(x_complex, ComplexTensor):
        real_imag = jt.stack([x_complex.real, x_complex.imag], dim=-1)
        return real_imag
    else:
        # x_complex已经是实数张量
        x_complex_np = x_complex.detach().numpy()
        real_imag = np.stack([x_complex_np.real, x_complex_np.imag], axis=-1)
        return jt.array(real_imag.astype(np.float32))

class ComplexTensor:
    """Jittor的复数张量表示"""
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def __mul__(self, other):
        if isinstance(other, ComplexTensor):
            # 复数乘法：(a+bi)(c+di) = (ac-bd) + (ad+bc)i
            real = self.real * other.real - self.imag * other.imag
            imag = self.real * other.imag + self.imag * other.real
            return ComplexTensor(real, imag)
        else:
            # 标量/张量乘法，并进行广播
            if hasattr(other, 'shape'):
                if len(other.shape) == 3 and len(self.real.shape) == 4:
                    # 添加批处理维度: [dim, H, W] -> [1, dim, H, W]
                    other = other.unsqueeze(0)
                elif len(other.shape) == len(self.real.shape) - 1:
                    # 添加前导维度
                    other = other.unsqueeze(0)
            return ComplexTensor(self.real * other, self.imag * other)

def rearrange(tensor, pattern):
    """使'b d h w ri -> b (d ri) h w'实现重新排列"""
    if pattern == 'b d h w ri -> b (d ri) h w':
        b, d, h, w, ri = tensor.shape
        return tensor.reshape(b, d * ri, h, w)
    else:
        raise NotImplementedError(f"Pattern {pattern} not implemented")

class Conditioning(nn.Module):
    """
    条件模块，相当于Dynamic Conditional Encoding
    param fmap_size: 特征图的尺寸
    param dim: 输入张量的通道数
    param dynamic: 是否使用动态条件注意力, 默认为True
    param image_size: 输入图像的尺寸, 默认为None
    param dim_head: 每个注意力头的维度, 默认为32
    param heads: 注意力头数, 默认为4
    param depth: Transformer的层数, 默认为4
    param patch_size: 分块大小, 默认为16
    """
    def __init__(
        self,
        fmap_size,
        dim,
        dynamic = True,
        image_size = None,
        dim_head = 32,
        heads = 4,
        depth = 4,
        patch_size = 16
    ):
        super().__init__()
        # 创建可学习的参数，相当于nn.Parameter(torch.ones(dim, fmap_size, fmap_size))
        # Jittor通过将张量分配给self属性来创建可学习的参数
        self.ff_parser_attn_map = jt.ones((dim, fmap_size, fmap_size), dtype=jt.float32)

        self.dynamic = dynamic

        if dynamic:
            self.to_dynamic_ff_parser_attn_map = ViT(
                dim = dim,
                channels = dim * 2 * 2,  # 输入和条件，考虑实部和虚部
                channels_out = dim,
                image_size = image_size,
                patch_size = patch_size,
                heads = heads,
                dim_head = dim_head,
                depth = depth
            )

        self.norm_input = LayerNorm(dim, bias = True)
        self.norm_condition = LayerNorm(dim, bias = True)

        self.block = ResnetBlock(dim, dim)

    # 前向传播
    def execute(self, x, c):
        ff_parser_attn_map = self.ff_parser_attn_map

        # ff-parser，用于调制高频噪声
        dtype = x.dtype
        x_complex = fft2(x)

        if self.dynamic:
            c_complex = fft2(c)
            x_as_real, c_as_real = view_as_real(x_complex), view_as_real(c_complex)
            x_as_real, c_as_real = rearrange(x_as_real, 'b d h w ri -> b (d ri) h w'), rearrange(c_as_real, 'b d h w ri -> b (d ri) h w')

            to_dynamic_input = jt.concat((x_as_real, c_as_real), dim = 1)

            dynamic_ff_parser_attn_map = self.to_dynamic_ff_parser_attn_map(to_dynamic_input)

            ff_parser_attn_map = ff_parser_attn_map + dynamic_ff_parser_attn_map

        # 将注意力图应用于复数张量               
        # ff_parser_attn_map: [dim, h, w] 或 [batch, dim, h, w]        
        # x_complex.real/imag: [batch, dim, h, w]

        if len(ff_parser_attn_map.shape) == 3:
            # 添加批处理维度: [dim, h, w] -> [1, dim, h, w]
            ff_parser_attn_map = ff_parser_attn_map.unsqueeze(0)

        # 将注意力图应用于实部和虚部
        x_real_filtered = x_complex.real * ff_parser_attn_map
        x_imag_filtered = x_complex.imag * ff_parser_attn_map
        x_complex_filtered = ComplexTensor(x_real_filtered, x_imag_filtered)

        x = ifft2(x_complex_filtered).real
        x = x.astype(dtype)

        # 论文中的公式3，Dynamic Conditional Encoding的关键方法
        normed_x = self.norm_input(x)
        normed_c = self.norm_condition(c)
        c = (normed_x * normed_c) * c

        # 最后添加一个下采样模块
        return self.block(c)