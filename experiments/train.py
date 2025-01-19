import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, get_linear_schedule_with_warmup
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import datetime
import argparse
import numpy as np

from src.models.sentiment_model import MultimodalSentimentModel, OptimizedMultimodalSentimentModel
from src.data.dataset import MultimodalDataset, collate_fn, get_balanced_sampler
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

def train_model(config, model, train_loader, val_loader):
    """训练模型并返回最佳模型状态和指标"""
    # 设置tensorboard
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', f'{config.strategy_name}_{current_time}')
    writer = SummaryWriter(log_dir)
    
    device = get_device()
    model = model.to(device)
    
    # 计算类别权重
    label_counts = torch.bincount(torch.tensor([batch['label'] for batch in train_loader.dataset]))
    weights = 1.0 / label_counts.float()
    weights = weights / weights.sum()
    weights = weights.to(device)
    
    # 可选的带权重的交叉熵损失
    criterion = nn.CrossEntropyLoss(weight=weights if config.use_weighted_loss else None, 
                                  label_smoothing=config.label_smoothing)
    
    # 分层设置学习率
    clip_backbone_params = []
    clip_head_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'clip' in name:
                if any(x in name for x in ['visual.proj', 'text.proj', 'resblocks.11']):
                    clip_head_params.append(param)
                else:
                    clip_backbone_params.append(param)
            else:
                other_params.append(param)
    
    # 优化器配置
    optimizer = optim.AdamW([
        {'params': clip_backbone_params, 'lr': config.learning_rate * 0.01},
        {'params': clip_head_params, 'lr': config.learning_rate * 0.1},
        {'params': other_params, 'lr': config.learning_rate}
    ], weight_decay=config.weight_decay)
    
    # 学习率调度器
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    best_val_f1 = 0
    best_model = None
    best_metrics = None
    no_improve = 0
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            optimizer.zero_grad()
            
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            # 记录训练信息
            writer.add_scalar('Loss/train_step', loss.item(),
                            epoch * len(train_loader) + batch_idx)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0],
                            epoch * len(train_loader) + batch_idx)
        
        # 计算训练指标
        train_loss = train_loss / len(train_loader)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        train_acc = accuracy_score(train_labels, train_preds)
        
        print(f"\nEpoch {epoch+1} Training Metrics:")
        print(f"Loss: {train_loss:.4f} | F1: {train_f1:.4f} | Acc: {train_acc:.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                texts = batch['text'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, texts)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_acc = accuracy_score(val_labels, val_preds)
        
        print("\nValidation Metrics:")
        print(f"Loss: {val_loss:.4f} | F1: {val_f1:.4f} | Acc: {val_acc:.4f}")
        
        # 记录每个epoch的指标
        writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        writer.add_scalars('F1 Score', {
            'train': train_f1,
            'val': val_f1
        }, epoch)
        writer.add_scalars('Accuracy', {
            'train': train_acc,
            'val': val_acc
        }, epoch)
        
        # 更新最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict().copy()
            best_metrics = {
                'epoch': epoch + 1,
                'val_f1': val_f1,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_f1': train_f1,
                'train_acc': train_acc,
                'train_loss': train_loss
            }
            no_improve = 0
        else:
            no_improve += 1
        
        # 早停
        if no_improve >= config.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    writer.close()
    return best_model, best_metrics

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion_type', type=str, default='concat', 
                       choices=['concat', 'attention', 'gated', 'bilinear', 'enhanced_concat'])
    
    # 添加新的参数
    parser.add_argument('--feature_adaptation', action='store_true', 
                       help='使用特征适配层')
    parser.add_argument('--unfreeze_layers', type=int, default=2,
                       help='解冻CLIP最后几层')
    
    # 原有的参数
    parser.add_argument('--use_weighted_loss', action='store_true')
    parser.add_argument('--no_weighted_loss', action='store_false', dest='use_weighted_loss')
    parser.add_argument('--use_balanced_sampler', action='store_true')
    parser.add_argument('--no_balanced_sampler', action='store_false', dest='use_balanced_sampler')
    parser.add_argument('--use_augmentation', action='store_true')
    parser.add_argument('--no_augmentation', action='store_false', dest='use_augmentation')
    parser.add_argument('--exp_name', type=str, default='')
    
    parser.set_defaults(
        use_weighted_loss=False,
        use_balanced_sampler=False,
        use_augmentation=False,
        feature_adaptation=False
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 配置初始化
    config = ModelConfig()
    config.fusion_type = args.fusion_type
    config.use_weighted_loss = args.use_weighted_loss
    config.use_balanced_sampler = args.use_balanced_sampler
    config.use_augmentation = args.use_augmentation
    config.feature_adaptation = args.feature_adaptation
    config.unfreeze_layers = args.unfreeze_layers
    
    # 设置实验名称
    if args.exp_name:
        config.strategy_name = args.exp_name
    else:
        config.strategy_name = f"{config.fusion_type}_{'w' if config.use_weighted_loss else 'n'}_{'b' if config.use_balanced_sampler else 'n'}_{'a' if config.use_augmentation else 'n'}"
    
    # 创建必要的目录
    os.makedirs('runs', exist_ok=True)  # 确保日志目录存在
    os.makedirs(config.checkpoint_dir, exist_ok=True)  # 确保模型保存目录存在
    
    # 初始化数据集和模型
    processor = CLIPProcessor.from_pretrained(config.clip_model_name)
    dataset = MultimodalDataset(
        config=config,
        processor=processor,
        annotation_file=config.train_file,
        augment=config.use_augmentation
    )
    
    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.cv_random_seed)
    )
    
    # 创建数据加载器
    if config.use_balanced_sampler:
        # 直接传递标签列表
        labels = [dataset.samples[i]['label'] for i in train_dataset.indices]
        train_sampler = get_balanced_sampler(labels)
        train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_dataset.indices),
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            collate_fn=collate_fn
        )
    
    val_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_dataset.indices),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    # 根据参数选择模型
    if config.feature_adaptation:
        model = OptimizedMultimodalSentimentModel(config)
        print("Using OptimizedMultimodalSentimentModel with feature adaptation")
    else:
        model = MultimodalSentimentModel(config)
        print("Using MultimodalSentimentModel")
    
    # 训练模型
    best_model_state, metrics = train_model(config, model, train_loader, val_loader)
    
    # 保存最佳模型
    save_path = os.path.join(config.checkpoint_dir, 
                            f'{config.strategy_name}_{datetime.datetime.now().strftime("%b%d_%H-%M-%S")}_best.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': best_model_state,
        'metrics': metrics,
        'config': vars(config)
    }, save_path)
    print(f"Saved best checkpoint to {save_path}")
    
    # 打印最终结果
    print("\nTraining completed!")
    print(f"Best epoch: {metrics['epoch']}")
    print(f"Validation F1: {metrics['val_f1']:.4f}")
    print(f"Validation Accuracy: {metrics['val_acc']:.4f}")
    print(f"Training F1: {metrics['train_f1']:.4f}")
    print(f"Training Accuracy: {metrics['train_acc']:.4f}")
    
    return metrics

if __name__ == '__main__':
    main()