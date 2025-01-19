import os

class ModelConfig:
    def __init__(self):
        # CLIP模型配置
        self.clip_model_name = "openai/clip-vit-base-patch32"  # 保持base版本，平衡性能和资源
        
        # 模型架构配置
        self.hidden_dim = 512  # 与CLIP base版本的输出维度匹配
        self.num_classes = 3   # 情感分类的类别数
        
        # 训练配置
        self.batch_size = 32   
        self.learning_rate = 2e-5  # 改回原来的学习率
        self.num_epochs = 20   
        self.weight_decay = 0.01  # 改回原来的权重衰减
        self.num_workers = 4   
        self.warmup_ratio = 0.1  # 改回原来的预热比例
        self.label_smoothing = 0.1  # 改回原来的标签平滑
        self.gradient_clip = 1.0  # 改回原来的梯度裁剪
        self.pin_memory = True
        
        # 学习率调度器配置（保留但简化）
        self.lr_scheduler_type = 'linear'  # 改用线性衰减
        self.lr_scheduler_factor = 0.1
        
        # 早停配置
        self.patience = 5  # 改回原来的早停耐心值
        self.min_delta = 0.001  # 改回原来的最小改善阈值
        
        # 添加新的训练稳定性配置
        self.warmup_steps = 100  # 固定步数的预热
        self.scheduler_steps = 1000  # 学习率调度的总步数
        self.max_grad_norm = 1.0  # 梯度裁剪的最大范数
        
        # 路径配置
        self.data_dir = "data/data"
        self.train_file = "data/train.txt"
        self.checkpoint_dir = "outputs/experiments"
        self.model_save_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        
        # 类别不平衡处理配置
        self.use_weighted_loss = True  # 启用加权损失
        self.use_balanced_sampler = False  # 与加权损失二选一
        self.use_augmentation = True  # 启用数据增强
        self.augment_minority_only = True  # 只增强少数类样本
        
        # 融合策略配置
        self.fusion_type = 'enhanced_concat'
        self.interaction_dim = 256  # 交互特征维度
        self.use_layer_norm = True
        
        # 交叉验证配置
        self.n_splits = 5  # 5折交叉验证
        self.cv_random_seed = 42  # 固定随机种子
        self.stratified = True  # 使用分层交叉验证
        self.shuffle_folds = True  # 打乱数据
        
        # 性能优化配置
        self.use_amp = True  # 使用混合精度训练
        self.eval_frequency = 1  # 每个epoch都验证
        
        # 特征提取配置
        self.feature_adaptation = True
        self.unfreeze_layers = 2  # 解冻最后几层
        self.adapter_dropout = 0.1
        