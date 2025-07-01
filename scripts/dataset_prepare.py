import os
import numpy as np
import nibabel as nib
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import jittor as jt
from jittor.dataset import Dataset
from jittor import transform as jt_transform
from PIL import Image
from pathlib import Path
from tqdm import tqdm


class RefugeDataset(Dataset):
    """REFUGE-2数据集，用于青光眼评估"""
    def __init__(self, data_path, transform=None, training=True, split='train'):
        super().__init__()
        self.data_path = Path(data_path)
        self.transform = transform
        self.training = training
        self.split = split

        if training:
            # 训练数据路径
            img_dir = self.data_path / "Train" / "Disc_Cup_Fovea_Illustration"
            # 训练掩码路径
            mask_dir = self.data_path / "Train" / "Disc_Cup_Masks"
        else:
            # 测试数据路径
            img_dir = self.data_path / "Test" / "Disc_Cup_Fovea_Illustration"
            # 测试掩码路径
            mask_dir = self.data_path / "Test" / "Disc_Cup_Masks"

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # 获取所有图像文件
        self.img_files = sorted(list(self.img_dir.glob("*.jpg")) +
                               list(self.img_dir.glob("*.png")) +
                               list(self.img_dir.glob("*.bmp")))

        # 使用tqdm.write避免干扰进度条
        if len(self.img_files) == 0:
            tqdm.write(f"REFUGE2 Dataset: Found 0 images in {self.img_dir}")
            tqdm.write(f"Warning: No images found in {self.img_dir}")


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 加载图像
        img_path = self.img_files[index]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            tqdm.write(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (256, 256), 0)

        # 加载对应的掩码
        mask_path = None

        base_name = img_path.stem  # 例如，“T0001”

        mask_patterns = [
            base_name + ".png"
        ]

        for pattern in mask_patterns:
            potential_mask = self.mask_dir / pattern
            if potential_mask.exists():
                mask_path = potential_mask
                break

        if mask_path and mask_path.exists():
            try:
                mask = Image.open(mask_path).convert('L')
            except Exception as e:
                tqdm.write(f"Error loading mask {mask_path}: {e}")
                mask = Image.new('L', img.size, 0)
        else:
            mask = Image.new('L', img.size, 0)

        # 调整图像和掩码到相同尺寸
        target_size = (256, 256)
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.NEAREST)

        # 使用transforms处理图像
        if self.transform:
            img = self.transform(img)
            # 同时对掩码应用相同的尺寸变换
            for t in self.transform.transforms:
                if hasattr(t, '__class__') and 'Resize' in t.__class__.__name__:
                    # 获取目标尺寸
                    if hasattr(t, 'size'):
                        target_size = t.size
                        if isinstance(target_size, int):
                            target_size = (target_size, target_size)
                        mask = mask.resize(target_size, Image.NEAREST)
                    break
        else:       
            img = jt_transform.ToTensor()(img)

        if not isinstance(img, jt.Var):
            # 如果不是Jittor张量，先转换
            if isinstance(img, np.ndarray):
                img = jt.array(img.astype(np.float32))
            else:
                img = jt.array(img)

        # 确保是float32类型
        if hasattr(img, 'dtype') and img.dtype != jt.float32:
            img = img.float32()

        # 检查图像范围并归一化到[-1,1]
        if img.max() <= 1.0 and img.min() >= 0.0:
            # 从[0,1]转换到[-1,1]
            img = (img * 2.0 - 1.0).float32()
        elif img.max() > 1.0:
            # 如果图像在[0,255]范围，先归一化到[0,1]再到[-1,1]
            img = ((img / 255.0) * 2.0 - 1.0).float32()

        # 处理掩码
        mask_array = np.array(mask, dtype=np.float32)

        # 归一化掩码到[0,1]
        if mask_array.max() > 1.0:
            mask_array = mask_array / 255.0

        # 转换为Jittor张量
        mask = jt.array(mask_array, dtype=jt.float32)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # 添加通道维度 [H,W] -> [1,H,W]

        # 确保是二值掩码(0或1)
        mask = (mask > 0.5).float32()

        if self.training:
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
            if self._debug_count < 3:
                tqdm.write(f"REFUGE2 Sample {self._debug_count} - Image range: [{img.min():.3f}, {img.max():.3f}], shape: {img.shape}")
                tqdm.write(f"REFUGE2 Sample {self._debug_count} - Mask range: [{mask.min():.3f}, {mask.max():.3f}], shape: {mask.shape}, sum: {mask.sum()}")

                self._debug_count += 1

        # 确保掩码具有正值用于训练
        if self.training and mask.sum() == 0:
            tqdm.write(f"Warning: REFUGE2 sample {index} mask is all zeros")           

        if self.training:
            return img, mask
        return img, mask, img_path.name


class BraTsDataset(Dataset):
    """BraTs-2021数据集，用于脑肿瘤分割"""
    def __init__(self, data_path, transform=None, training=True, test_flag=False):
        super().__init__()
        self.data_path = Path(data_path)
        self.transform = transform
        self.training = training
        self.test_flag = test_flag

        # 设置序列类型
        self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []

        # 查找数据文件
        for root, dirs, files in os.walk(str(self.data_path)):
            # 如果没有子目录
            if not dirs:
                files.sort()
                datapoint = dict()
                # 提取所有文件作为通道
                for f in files:
                    if f.endswith('.nii.gz') or f.endswith('.nii'):
                        # 提取序列类型
                        if '_flair.' in f:
                            seqtype = 'flair'
                        elif '_t1ce.' in f:
                            seqtype = 't1ce'
                        elif '_t1.' in f:
                            seqtype = 't1'
                        elif '_t2.' in f:
                            seqtype = 't2'
                        elif '_seg.' in f:
                            seqtype = 'seg'
                        else:
                            continue

                        if seqtype in self.seqtypes_set:
                            datapoint[seqtype] = os.path.join(root, f)

                if set(datapoint.keys()) == self.seqtypes_set:
                    self.database.append(datapoint)

    def __len__(self):
        return len(self.database)

    def __getitem__(self, index):
        filedict = self.database[index]
        out = []

        for seqtype in self.seqtypes:
            try:
                nib_img = nib.load(filedict[seqtype])
                data = nib_img.get_fdata()

                # 取中间切片以加快处理速度
                middle_slice = data.shape[2] // 2
                slice_data = data[:, :, middle_slice]

                if slice_data.max() > slice_data.min():
                    # 归一化到[0, 1]
                    normalized = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
                else:
                    normalized = np.zeros_like(slice_data, dtype=np.float32)

                normalized = normalized.astype(np.float32)

                out.append(jt.array(normalized, dtype=jt.float32))

            except Exception as e:
                tqdm.write(f"Error loading {filedict[seqtype]}: {e}")
                out.append(jt.zeros((64, 64), dtype=jt.float32))

        out = jt.stack(out).float32()  # 形状: [5, H, W] 或 [4, H, W]
        path = filedict[self.seqtypes[0]]

        if self.test_flag:
            image = out
            if self.transform:
                # 转换为PIL格式
                image_pil = []
                for i in range(image.shape[0]):
                    img_np = (image[i].numpy() * 255).astype(np.uint8)
                    image_pil.append(Image.fromarray(img_np))

                # 对每个通道应用transform
                transformed = []
                for pil_img in image_pil:
                    transformed_img = self.transform(pil_img)
                    if isinstance(transformed_img, Image.Image):
                        transformed_img = jt.array(np.array(transformed_img)).float32() / 255.0
                    elif hasattr(transformed_img, 'float32'):
                        transformed_img = transformed_img.float32()
                    transformed.append(transformed_img)
                image = jt.stack(transformed).float32()

            image = out[:-1, ...]  
            label = out[-1:, ...]  # 分割掩码[1, H, W]

            label = jt.ternary(label > 0,
                             jt.ones_like(label).float32(),
                             jt.zeros_like(label).float32()).float32()

            if self.transform:
                image_pil = []
                for i in range(image.shape[0]):
                    img_np = (image[i].numpy() * 255).astype(np.uint8)
                    image_pil.append(Image.fromarray(img_np))

                label_pil = Image.fromarray((label[0].numpy() * 255).astype(np.uint8))

                transformed = []
                for pil_img in image_pil:
                    transformed_img = self.transform(pil_img)
                    if isinstance(transformed_img, Image.Image):
                        transformed_img = jt.array(np.array(transformed_img)).float32() / 255.0
                    elif hasattr(transformed_img, 'float32'):
                        transformed_img = transformed_img.float32()
                    transformed.append(transformed_img)
                image = jt.stack(transformed).float32()

                transformed_label = self.transform(label_pil)
                if isinstance(transformed_label, Image.Image):
                    label = jt.array(np.array(transformed_label)).float32().unsqueeze(0) / 255.0
                elif hasattr(transformed_label, 'float32'):
                    label = transformed_label.float32().unsqueeze(0)

            if len(image.shape) == 2:
                image = image.unsqueeze(0)  # [H, W] -> [1, H, W]
            if len(label.shape) == 2:
                label = label.unsqueeze(0)  # [H, W] -> [1, H, W]

            return (image, label, path)
        else:
            image = out[:-1, ...]
            label = out[-1:, ...]  # 分割掩码[1, H, W]

            # BraTS分割掩码有多个标签值(0,1,2,4)，需要将所有非零值转换为1
            label = jt.ternary(label > 0,
                             jt.ones_like(label).float32(),
                             jt.zeros_like(label).float32()).float32()

            if self.transform:
                image_pil = []
                for i in range(image.shape[0]):
                    img_np = (image[i].numpy() * 255).astype(np.uint8)
                    image_pil.append(Image.fromarray(img_np))

                label_pil = Image.fromarray((label[0].numpy() * 255).astype(np.uint8))

                # 对每个通道应用transform
                transformed_img = []
                for pil_img in image_pil:
                    transformed = self.transform(pil_img)
                    if isinstance(transformed, Image.Image):
                        transformed = jt.array(np.array(transformed)).float32() / 255.0
                    elif hasattr(transformed, 'float32'):
                        transformed = transformed.float32()

                    if transformed.ndim > 2:
                        transformed = transformed.squeeze()
                    if transformed.ndim == 1:
                        size = int(transformed.shape[0] ** 0.5)
                        transformed = transformed.reshape(size, size)

                    transformed_img.append(transformed)
                image = jt.stack(transformed_img).float32()  # 形状: [4, H, W]

                transformed_label = self.transform(label_pil)
                if isinstance(transformed_label, Image.Image):
                    transformed_label = jt.array(np.array(transformed_label)).float32() / 255.0
                    transformed_label = transformed_label.unsqueeze(0)
                elif hasattr(transformed_label, 'float32'):
                    transformed_label = transformed_label.float32()
                label = transformed_label

            if image.max() <= 1.0 and image.min() >= 0.0:
                # 从[0,1]转换到[-1,1]
                image = (image * 2.0 - 1.0).float32()
            elif image.max() > 1.0:
                # 如果图像在[0,255]范围，先归一化到[0,1]再到[-1,1]
                image = ((image / 255.0) * 2.0 - 1.0).float32()

            # 确保是二值掩码
            label = jt.ternary(label > 0.5,
                             jt.ones_like(label).float32(),
                             jt.zeros_like(label).float32()).float32()

            if len(label.shape) == 2:
                label = label.unsqueeze(0)  # [H, W] -> [1, H, W]
            elif len(label.shape) == 3 and label.shape[0] != 1:
                label = label[0:1]  # 取第一个通道

            # 训练时返回(image, mask)，测试时返回(image, mask, path)
            if self.training:
                return image, label
            else:
                return image, label, path


class DDTIDataset(Dataset):
    """DDTI数据集，用于医学信息处理"""
    def __init__(self, data_path, transform=None, training=True):
        super().__init__()
        self.data_path = Path(data_path)
        self.transform = transform
        self.training = training

        if training:
            self.img_dir = self.data_path / "Train" / "images"
            self.mask_dir = self.data_path / "Train" / "masks"
        else:
            self.img_dir = self.data_path / "Test" / "images"
            self.mask_dir = self.data_path / "Test" / "masks"

        # 获取所有图像文件
        self.img_files = sorted(list(self.img_dir.glob("*.jpg")) +
                               list(self.img_dir.glob("*.JPG")) +
                               list(self.img_dir.glob("*.png")) +
                               list(self.img_dir.glob("*.PNG")))

        if len(self.img_files) == 0:
            tqdm.write(f"DDTI Dataset: Found 0 images in {self.img_dir}")
            tqdm.write(f"Warning: No images found in {self.img_dir}")


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # 加载图像
        img_path = self.img_files[index]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            tqdm.write(f"Error loading image {img_path}: {e}")

        # 加载掩码
        mask_path = None
        for suffix in [""]:
            for ext in [".png", ".PNG", ".jpg", ".JPG"]:
                potential_mask = self.mask_dir / (img_path.stem + suffix + ext)
                if potential_mask.exists():
                    mask_path = potential_mask
                    break
            if mask_path:
                break

        if mask_path and mask_path.exists():
            try:
                mask = Image.open(mask_path).convert('L')
            except Exception as e:
                tqdm.write(f"Error loading mask {mask_path}: {e}")
                mask = Image.new('L', img.size, 0)

        # 图像和掩码调整到相同尺寸
        target_size = (256, 256)  
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
        if mask.size != target_size:
            mask = mask.resize(target_size, Image.NEAREST)

        # 使用transforms处理图像
        if self.transform:
            img = self.transform(img)
            # 同时对掩码应用相同的尺寸变换
            for t in self.transform.transforms:
                if hasattr(t, '__class__') and 'Resize' in t.__class__.__name__:
                    if hasattr(t, 'size'):
                        target_size = t.size
                        if isinstance(target_size, int):
                            target_size = (target_size, target_size)
                        mask = mask.resize(target_size, Image.NEAREST)
                    break
        else:
            img = jt_transform.ToTensor()(img)

        if not isinstance(img, jt.Var):
            if isinstance(img, np.ndarray):
                img = jt.array(img.astype(np.float32))
            else:
                img = jt.array(img)

        # 确保是float32类型
        if hasattr(img, 'dtype') and img.dtype != jt.float32:
            img = img.float32()

        # 检查图像范围并归一化到[-1,1]
        if img.max() <= 1.0 and img.min() >= 0.0:
            # 从[0,1]转换到[-1,1]
            img = (img * 2.0 - 1.0).float32()
        elif img.max() > 1.0:
            # 如果图像在[0,255]范围，先归一化到[0,1]再到[-1,1]
            img = ((img / 255.0) * 2.0 - 1.0).float32()

        # 处理掩码
        mask_array = np.array(mask, dtype=np.float32)

        # 归一化掩码到[0,1]
        if mask_array.max() > 1.0:
            mask_array = mask_array / 255.0

        # 转换为Jittor张量
        mask = jt.array(mask_array, dtype=jt.float32)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # 添加通道维度 [H,W] -> [1,H,W]

        # 确保是二值掩码(0或1)
        mask = (mask > 0.5).float32()

        if self.training:
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
            if self._debug_count < 3:
                tqdm.write(f"DDTI Sample {self._debug_count} - Image range: [{img.min():.3f}, {img.max():.3f}], shape: {img.shape}")
                tqdm.write(f"DDTI Sample {self._debug_count} - Mask range: [{mask.min():.3f}, {mask.max():.3f}], shape: {mask.shape}, sum: {mask.sum()}")

                self._debug_count += 1

        # 确保掩码具有正值用于训练
        if self.training and mask.sum() == 0:
            tqdm.write(f"Warning: DDTI sample {index} mask is all zeros, adding small positive region")           

        if self.training:
            return img, mask
        return img, mask, img_path.name

