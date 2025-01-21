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
    config = ModelConfig()
    config.fusion_type = 'bilinear'  
    config.use_balanced_sampler = True  
    config.use_augmentation = True  
    
    checkpoint_path = 'outputs/experiments/ablation_both_bilinear_Jan21_14-27-57_best.pth' # 更改为你的最佳模型路径
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    
    model = MultimodalSentimentModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = get_device()
    model = model.to(device)
    model.eval()
    
    processor = CLIPProcessor.from_pretrained(config.clip_model_name)
    test_dataset = MultimodalDataset(
        config=config,
        processor=processor,
        annotation_file='data/test_without_label.txt' 
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    test_guids = []
    with open('data/test_without_label.txt', 'r') as f:
        next(f) 
        for line in f:
            guid = line.strip().split(',')[0]
            test_guids.append(guid)
    
    print("\nGenerating predictions...")
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            outputs = model(images, texts)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    output_file = 'predictions.txt'
    print(f"\nSaving predictions to {output_file}")
    
    id_to_label = {
        0: 'negative',
        1: 'neutral',
        2: 'positive'
    }
    
    with open(output_file, 'w') as f:
        f.write('guid,tag\n') 
        for guid, pred in zip(test_guids, predictions):
            label = id_to_label[pred]
            f.write(f'{guid},{label}\n')
    
    print("Prediction completed!")

if __name__ == '__main__':
    predict() 