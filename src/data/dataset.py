import os
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
from src.data.processor import MultimodalPreprocessor
import chardet
from torchvision import transforms
import random
import numpy as np

def load_text(file_path):
    """安全加载文本文件，尝试多种编码方式"""
    encodings = ['utf-8', 'gb2312', 'gbk', 'big5', 'euc-jp', 'shift-jis']
    
    # 1. 首先尝试常见编码
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read().strip()
                if text:
                    return text
        except:
            continue
    
    # 2. 如果常见编码都失败，使用二进制读取并尝试检测编码
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        # 使用chardet检测编码
        result = chardet.detect(raw_data)
        if result['encoding']:
            try:
                text = raw_data.decode(result['encoding']).strip()
                if text:
                    return text
            except:
                pass
    except:
        pass
    
    # 3. 如果所有方法都失败，尝试直接解码字节
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        # 忽略无法解码的字符
        text = raw_data.decode('utf-8', errors='ignore').strip()
        if text:
            return text
    except:
        pass
    
    # 4. 如果完全无法读取，返回默认文本
    return "无法读取的文本"

class MultimodalDataset(Dataset):
    def __init__(self, config, processor, annotation_file, augment=False):
        self.data_dir = config.data_dir
        self.processor = MultimodalPreprocessor(processor)
        self.transform = None
        self.augment = augment
        self.aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
        ])
        
        # 读取标签文件
        self.samples = []
        with open(annotation_file, 'r', encoding='utf-8') as f:
            # 跳过标题行
            next(f)
            for line in f:
                guid, label = line.strip().split(',')
                if label.strip():  # 训练集有标签
                    # 加载文本
                    text_path = os.path.join(self.data_dir, f"{guid}.txt")
                    text = load_text(text_path)
                    
                    self.samples.append({
                        'guid': guid,
                        'text': text,
                        'label': self.label_to_id(label)
                    })
                else:  # 测试集无标签
                    text_path = os.path.join(self.data_dir, f"{guid}.txt")
                    text = load_text(text_path)
                    
                    self.samples.append({
                        'guid': guid,
                        'text': text,
                        'label': -1
                    })
    
    def label_to_id(self, label):
        label_map = {
            'positive': 0,
            'neutral': 1,
            'negative': 2
        }
        return label_map.get(label, -1)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 使用字典访问
        sample = self.samples[idx]
        guid = sample['guid']
        text = sample['text']
        label = sample['label']
        
        # 加载图像
        image_path = os.path.join(self.data_dir, f"{guid}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        # 对少数类进行数据增强
        if self.augment and label != 2:  # 非正面情感样本进行增强
            if random.random() < 0.5:  # 50%的概率进行增强
                image = self.aug_transforms(image)
        
        # 处理图像和文本
        try:
            inputs = self.processor(
                text=text,
                image=image
            )
        except Exception as e:
            print(f"Error processing data for guid {guid}: {e}")
            raise
        
        return {
            'guid': guid,
            'image': inputs['pixel_values'],
            'text': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    """
    自定义 collate 函数，确保批次中的数据可以正确堆叠
    """
    # 获取批次中的最大长度
    max_text_length = max(item['text'].size(0) for item in batch)
    
    # 初始化批次数据
    batch_size = len(batch)
    image_tensors = torch.stack([item['image'] for item in batch])
    text_tensors = torch.zeros((batch_size, max_text_length), dtype=torch.long)
    attention_masks = torch.zeros((batch_size, max_text_length), dtype=torch.long)
    labels = torch.stack([item['label'] for item in batch])
    guids = [item['guid'] for item in batch]
    
    # 填充文本数据
    for i, item in enumerate(batch):
        text_length = item['text'].size(0)
        text_tensors[i, :text_length] = item['text']
        attention_masks[i, :text_length] = item['attention_mask']
    
    return {
        'guid': guids,
        'image': image_tensors,
        'text': text_tensors,
        'attention_mask': attention_masks,
        'label': labels
    }

def get_balanced_sampler(labels):
    """创建平衡采样器
    
    Args:
        labels: 标签列表或包含标签的数据集
    """
    if isinstance(labels, list):
        # 如果直接传入标签列表
        label_list = labels
    elif isinstance(labels, torch.utils.data.Subset):
        # 如果是数据集的子集
        label_list = [labels.dataset.samples[i]['label'] for i in labels.indices]
    else:
        # 如果是完整数据集
        label_list = [sample['label'] for sample in labels.samples]
    
    # 计算类别权重
    label_counts = torch.bincount(torch.tensor(label_list))
    weights = 1.0 / label_counts.float()
    weights = weights / weights.sum()
    
    # 为每个样本分配权重
    sample_weights = [weights[label] for label in label_list]
    
    # 创建采样器
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler