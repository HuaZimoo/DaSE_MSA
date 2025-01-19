import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
import os

from src.models.sentiment_model import MultimodalSentimentModel
from src.data.dataset import MultimodalDataset
from configs.model_config import ModelConfig

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def predict(config, model, test_loader, device):
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for batch in test_loader:
            guids = batch['guid']
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            
            outputs = model(images, texts)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for guid, pred in zip(guids, preds):
                predictions[guid] = pred
    
    return predictions

def save_predictions(predictions, output_file):
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    id_to_label = {
        0: 'positive',
        1: 'neutral',
        2: 'negative'
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('guid,sentiment\n')
        for guid, pred in predictions.items():
            f.write(f'{guid},{id_to_label[pred]}\n')

def main():
    config = ModelConfig()
    device = get_device()
    
    # 初始化处理器和模型
    processor = CLIPProcessor.from_pretrained(config.clip_model_name)
    model = MultimodalSentimentModel(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path))
    
    # 加载测试数据集
    test_dataset = MultimodalDataset(
        config=config,
        processor=processor,
        annotation_file=config.test_file
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 预测并保存结果
    predictions = predict(config, model, test_loader, device)
    save_predictions(predictions, config.prediction_save_path)

if __name__ == '__main__':
    main() 