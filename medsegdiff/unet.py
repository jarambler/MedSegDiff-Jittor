import copy
from functools import partial
import jittor as jt
from jittor import nn
from .utils import default, SinusoidalPosEmb, Downsample, Upsample
from .blocks import ResnetBlock
from .attention import LinearAttention, Attention, Transformer
from .conditioning import Conditioning

class Unet(nn.Module):
    """
    U-Net架构
    param dim: 输入张量的通道数
    param image_size: 输入图像的尺寸
    param mask_channels: 掩码通道数, 默认为1
    param input_img_channels: 输入图像的通道数, 默认为3
    param init_dim: 初始卷积层的输出通道数, 默认为None
    param out_dim: 输出卷积层的输入通道数, 默认为None
    param dim_mults: 特征图通道数的倍数, 默认为(1, 2, 4, 8)
    param full_self_attn: 是否在每个downsample层使用自注意力, 默认为(1, 1, 1, 0)
    param attn_heads: 注意力头数, 默认为4
    param attn_dim_head: 每个注意力头的维度, 默认为32
    param mid_transformer_depth: 中间Transformer的层数, 默认为1
    param self_condition: 是否使用自条件注意力, 默认为False
    param resnet_block_groups: ResNet块的组数, 默认为8
    param conditioning_klass: 条件模块的类, 默认为Conditioning
    param skip_connect_condition_famps: 是否使用Dynamic Conditional Encoding, 默认为False
    param dynamic_ff_parser_attn_map: 是否使用FF-Parser, 默认为False
    param conditioning_kwargs: 条件模块的关键字参数, 默认为dict(heads=4, dim_head=32, depth=4, patch_size=16)
    """
    def __init__(
            self,
            dim,
            image_size,
            mask_channels=1,
            input_img_channels=3,
            init_dim=None,
            out_dim=None,
            dim_mults: tuple = (1, 2, 4, 8),
            full_self_attn: tuple = (False, False, False, True),
            attn_heads=4,
            attn_dim_head=32,
            mid_transformer_depth=1,
            self_condition=False,
            resnet_block_groups=8,
            conditioning_klass=Conditioning,
            skip_connect_condition_famps=False,
            dynamic_ff_parser_attn_map=False,
            conditioning_kwargs: dict = dict(
                heads=2,
                dim_head=16,
                depth=2,
                patch_size=16
            )
    ):
        super().__init__()

        self.image_size = image_size
        self.input_img_channels = input_img_channels
        self.mask_channels = mask_channels
        self.self_condition = self_condition
        
        # 输出通道数处理
        output_channels = mask_channels
        # 掩码通道数翻倍
        mask_channels = mask_channels * (2 if self_condition else 1)

        # 初始化卷积层
        init_dim = default(init_dim, dim)
        # 掩码通过卷积映射到init_dim
        self.init_conv = nn.Conv2d(mask_channels, init_dim, 7, padding=3)
        # 条件图像通过卷积映射到init_dim
        self.cond_init_conv = nn.Conv2d(input_img_channels, init_dim, 7, padding=3)

        # 每层的通道数列表
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # ResNet块
        block_klass = partial(
            ResnetBlock, 
            groups=resnet_block_groups
        )

        # 时间嵌入
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 注意力参数
        attn_kwargs = dict(
            dim_head=attn_dim_head,
            heads=attn_heads
        )

        # 条件模块
        if conditioning_klass == Conditioning:
            conditioning_klass = partial(
                Conditioning,
                dynamic=dynamic_ff_parser_attn_map,
                **conditioning_kwargs
            )

        num_resolutions = len(in_out)
        assert len(full_self_attn) == num_resolutions
        
        # 下采样层与Dynamic Conditional Encoding
        self.conditioners = nn.ModuleList([])
        self.skip_connect_condition_famps = skip_connect_condition_famps       
        self.downs = nn.ModuleList([])

        curr_fmap_size = image_size

        # 创建下采样层和Dynamic Conditional Encoding
        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(in_out, full_self_attn)):
            is_last = ind >= (num_resolutions - 1)
            attn_klass = Attention if full_attn else LinearAttention

            # 每个下采样层都有一个Dynamic Conditional Encoding
            self.conditioners.append(
                conditioning_klass(
                    curr_fmap_size,
                    dim_in,
                    image_size=curr_fmap_size
                )
            )

            # 下采样层：两个ResNet块 + 注意力 + 下采样/卷积
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                attn_klass(dim_in, **attn_kwargs),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))
            
            if not is_last:
                curr_fmap_size //= 2

        # 中间层：两个ResNet块 + Transformer + 两个ResNet块
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_transformer = Transformer(mid_dim, depth=mid_transformer_depth, **attn_kwargs)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # 条件下采样模块
        self.cond_downs = copy.deepcopy(self.downs)
        self.cond_mid_block1 = copy.deepcopy(self.mid_block1)

        # 上采样模块
        self.ups = nn.ModuleList([])

        # 创建上采样模块
        for ind, ((dim_in, dim_out), full_attn) in enumerate(zip(reversed(in_out), reversed(full_self_attn))):
            is_last = ind == (len(in_out) - 1)
            attn_klass = Attention if full_attn else LinearAttention

            # 如果使用skip连接，concat的维度翻倍
            skip_connect_dim = dim_in * (2 if self.skip_connect_condition_famps else 1)

            # 上采样层：两个ResNet块 + 注意力 + 上采样/卷积
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + skip_connect_dim, dim_out, time_emb_dim=time_dim),
                attn_klass(dim_out, **attn_kwargs),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        # 最终的ResNet块
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, output_channels, 1)

    # 前向传播
    def execute(
            self,
            x,
            time,
            condition,
            x_self_cond=None
    ):
        dtype, skip_connect_condition_famps = x.dtype, self.skip_connect_condition_famps

        # 如果使用自条件注意力，将掩码和自条件注意力连接起来
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: jt.zeros_like(x))
            x = jt.concat((x_self_cond, x), dim=1)

        # 输入图像与条件图像的初始卷积
        x = self.init_conv(x)
        r = x.clone()    # 保存初始特征图用于skip connection
        c = self.cond_init_conv(condition)

        # 时间嵌入
        t = self.time_mlp(time)

        hiddens = []

        # 编码路径
        for (block1, block2, attn, downsample), (cond_block1, cond_block2, cond_attn, cond_downsample), conditioner in zip(self.downs, self.cond_downs, self.conditioners):
            x = block1(x, t)
            c = cond_block1(c, t)

            # 保存特征图用于skip connection
            hiddens.append([x, c] if skip_connect_condition_famps else [x])

            x = block2(x, t)
            c = cond_block2(c, t)

            # 条件融合，即Dynamic Conditional Encoding
            c = conditioner(x, c)

            # 保存特征图用于skip connection
            hiddens.append([x, c] if skip_connect_condition_famps else [x])

            x = downsample(x)
            c = cond_downsample(c)

        # 中间Transformer
        x = self.mid_block1(x, t)
        c = self.cond_mid_block1(c, t)

        # 条件图与主图融合(残差连接)
        x = x + c

        x  = self.mid_transformer(x)
        x = self.mid_block2(x, t)

        # 解码路径
        for block1, block2, attn, upsample in self.ups:
            x = jt.concat((x, *hiddens.pop()), dim=1)
            x = block1(x, t)

            x = jt.concat((x, *hiddens.pop()), dim=1)
            x = block2(x, t)

            x = attn(x)

            x = upsample(x)

        # 最后的ResNet块
        x = jt.concat((x, r), dim=1)
        x = self.final_res_block(x, t)
        
        return self.final_conv(x)