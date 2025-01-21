import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from src.models.sentiment_model import MultimodalSentimentModel
from src.data.dataset import MultimodalDataset, collate_fn
from configs.model_config import ModelConfig

def get_device():
    """获取可用的计算设备"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def predict():
    # 配置初始化
    config = ModelConfig()
    config.fusion_type = 'bilinear'  # 使用双线性融合
    config.use_balanced_sampler = True  # 保持与训练时一致
    config.use_augmentation = True  # 保持与训练时一致
    
    # 加载最佳模型（添加weights_only=True）
    checkpoint_path = 'outputs/experiments/ablation_text_only_bilinear_Jan21_15-06-20_best.pth'
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    
    model = MultimodalSentimentModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # 准备数据（移除is_test参数）
    processor = CLIPProcessor.from_pretrained(config.clip_model_name)
    test_dataset = MultimodalDataset(
        config=config,
        processor=processor,
        annotation_file='data/test_without_label.txt'  # 移除is_test参数
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    # 读取测试集的GUID
    test_guids = []
    with open('data/test_without_label.txt', 'r') as f:
        next(f)  # 跳过header行
        for line in f:
            guid = line.strip().split(',')[0]
            test_guids.append(guid)
    
    # 进行预测
    print("\nGenerating predictions...")
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            outputs = model(images, texts)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    # 保存预测结果
    output_file = 'predictions.txt'
    print(f"\nSaving predictions to {output_file}")
    
    # 数字标签转换为文本标签
    id_to_label = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }
    
    # 按照原格式写入结果
    with open(output_file, 'w') as f:
        f.write('guid,tag\n')  # 写入header
        for guid, pred in zip(test_guids, predictions):
            label = id_to_label[pred]
            f.write(f'{guid},{label}\n')
    
    print("Prediction completed!")

if __name__ == '__main__':
    predict() 