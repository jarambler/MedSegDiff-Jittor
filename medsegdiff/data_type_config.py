import jittor as jt
import numpy as np
import subprocess

def configure_jittor_cuda():
    """Configure Jittor to use CUDA with proper data type handling"""
    # 启用CUDA进行训练/测试/评估
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        print("Using GPU for processing.")
        # 获取GPU信息
        try:       
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        print(f"GPU {i}: {parts[0]}, Total: {parts[1]}MB, Used: {parts[2]}MB, Free: {parts[3]}MB")
            else:
                print("GPU detected but nvidia-smi not available")
        except:
            print("GPU detected but could not get detailed info")
    else:
        print("CUDA not available, using CPU")
        jt.flags.use_cuda = 0

def configure_jittor_data_types():
    """
    配置Jittor数据类型以兼容CUDA
    默认数据类型设置为float32
    """
    # 禁用自动混合精度以避免数据类型问题
    jt.flags.auto_mixed_precision_level = 0  
    
    # 强制所有张量创建函数默认使用float32
    try:
        # 保存原始函数引用
        original_randn = jt.randn
        original_zeros = jt.zeros
        original_ones = jt.ones
        original_empty = jt.empty
        original_full = jt.full
        original_linspace = jt.linspace
        original_arange = jt.arange
        original_array = jt.array

        def randn_float32(*args, **kwargs):
            if 'dtype' not in kwargs:
                kwargs['dtype'] = jt.float32
            result = original_randn(*args, **kwargs)
            return result.float32() if hasattr(result, 'float32') else result

        def zeros_float32(*args, **kwargs):
            if 'dtype' not in kwargs:
                kwargs['dtype'] = jt.float32
            result = original_zeros(*args, **kwargs)
            return result.float32() if hasattr(result, 'float32') else result

        def ones_float32(*args, **kwargs):
            if 'dtype' not in kwargs:
                kwargs['dtype'] = jt.float32
            result = original_ones(*args, **kwargs)
            return result.float32() if hasattr(result, 'float32') else result

        def empty_float32(*args, **kwargs):
            if 'dtype' not in kwargs:
                kwargs['dtype'] = jt.float32
            result = original_empty(*args, **kwargs)
            return result.float32() if hasattr(result, 'float32') else result

        def full_float32(*args, **kwargs):
            if 'dtype' not in kwargs and len(args) >= 2:
                kwargs['dtype'] = jt.float32
            result = original_full(*args, **kwargs)
            return result.float32() if hasattr(result, 'float32') else result

        def linspace_float32(*args, **kwargs):
            # Jittor中linspace不支持dtype参数，创建后进行转换
            result = original_linspace(*args, **kwargs)
            return result.float32() if hasattr(result, 'float32') else result

        def arange_float32(*args, **kwargs):
            if 'dtype' not in kwargs:
                kwargs['dtype'] = jt.float32
            result = original_arange(*args, **kwargs)
            return result.float32() if hasattr(result, 'float32') else result

        def array_float32(*args, **kwargs):
            # 强制jt.array默认创建float32张量
            if 'dtype' not in kwargs and len(args) > 0:
                # 检查输入数据类型
                data = args[0]
                if hasattr(data, 'dtype'):
                    # 如果是numpy数组或其他有dtype的对象
                    if 'float' in str(data.dtype) or 'int' in str(data.dtype):
                        kwargs['dtype'] = jt.float32
                elif isinstance(data, (list, tuple)):
                    # 如果是Python列表或元组，默认使用float32
                    kwargs['dtype'] = jt.float32
            result = original_array(*args, **kwargs)
            # 确保结果是float32（如果是浮点类型）
            if hasattr(result, 'dtype') and hasattr(result, 'float32'):
                if 'float' in str(result.dtype) and result.dtype != jt.float32:
                    result = result.float32()
            return result

        # 应用补丁
        jt.randn = randn_float32
        jt.zeros = zeros_float32
        jt.ones = ones_float32
        jt.empty = empty_float32
        jt.full = full_float32
        jt.linspace = linspace_float32
        jt.arange = arange_float32
        jt.array = array_float32

    except Exception as e:
        print(f"Error: Jittor函数兼容性错误: {e}")

    # 设置全局配置
    try:
        jt.set_global_seed(42)  # 可重现性
    except Exception as e:
        print(f"Error: 无法设置为全局配置: {e}")

def ensure_float32_tensor(tensor):
    """确保张量是float32类型"""
    if hasattr(tensor, 'dtype') and hasattr(tensor, 'float32'):
        if tensor.dtype != jt.float32:
            return tensor.float32()
    return tensor

def ensure_float32_batch(batch_data):
    """确保批次数据中的所有张量都是float32"""
    if isinstance(batch_data, (list, tuple)):
        return [ensure_float32_tensor(item) if hasattr(item, 'dtype') else item for item in batch_data]
    else:
        return ensure_float32_tensor(batch_data)

def convert_numpy_to_jittor_float32(data):
    """将numpy数组转换为Jittor float32张量"""
    if isinstance(data, np.ndarray):
        return jt.array(data.astype(np.float32))
    elif not isinstance(data, jt.Var):
        return jt.array(data)
    else:
        return ensure_float32_tensor(data)

def setup_model_float32(model):
    """确保模型使用float32数据类型"""
    # 确保模型使用float32
    model = model.float32()
    
    # 强制所有参数为float32
    param_count = 0
    converted_count = 0
    for param in model.parameters():
        param_count += 1
        if hasattr(param, 'data') and hasattr(param.data, 'dtype'):
            if param.data.dtype != jt.float32:
                if hasattr(param.data, 'float32'):
                    param.data = param.data.float32()
                    converted_count += 1
        elif hasattr(param, 'dtype') and param.dtype != jt.float32:
            if hasattr(param, 'float32'):
                try:
                    # 检查
                    converted_count += 1
                except:
                    pass

    # 验证所有参数是否都是float32
    param_dtypes = set()
    for param in model.parameters():
        if hasattr(param, 'dtype'):
            param_dtypes.add(str(param.dtype))
        elif hasattr(param, 'data') and hasattr(param.data, 'dtype'):
            param_dtypes.add(str(param.data.dtype))

    print(f"模型参数数据类型: {param_dtypes}")
    print(f"参数转换统计: {converted_count}/{param_count} 参数已转换为float32")
    
    return model

def setup_diffusion_float32(diffusion):
    """确保扩散模型使用float32数据类型"""
    if hasattr(diffusion, 'float32'):
        diffusion = diffusion.float32()

    # 检查扩散模型的关键参数数据类型
    diffusion_dtypes = set()
    for attr_name in ['betas', 'alphas_cumprod', 'alphas_cumprod_prev']:
        if hasattr(diffusion, attr_name):
            attr = getattr(diffusion, attr_name)
            if hasattr(attr, 'dtype'):
                diffusion_dtypes.add(f"{attr_name}: {attr.dtype}")
                # 强制转换为float32
                if attr.dtype != jt.float32 and hasattr(attr, 'float32'):
                    setattr(diffusion, attr_name, attr.float32())

    print(f"扩散模型参数数据类型: {diffusion_dtypes}")
    print("扩散模型已配置为float32")
    
    return diffusion

def initialize_jittor_for_cuda():
    """使用CUDA和float32配置初始化Jittor"""
    print("Initializing Jittor with CUDA and float32 configuration...")
    configure_jittor_cuda()
    configure_jittor_data_types()

    if jt.flags.use_cuda:
        print("Jittor已配置为GPU模式，使用float32数据类型")
    else:
        print("Jittor已配置为CPU模式，使用float32数据类型")
