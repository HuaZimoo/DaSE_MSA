import os

class ModelConfig:
    def __init__(self):
        self.clip_model_name = "openai/clip-vit-base-patch32"  
        
        self.hidden_dim = 512  
        self.num_classes = 3  
        
        self.batch_size = 32   
        self.learning_rate = 2e-5 
        self.num_epochs = 20   
        self.weight_decay = 0.01 
        self.num_workers = 4   
        self.warmup_ratio = 0.1 
        self.label_smoothing = 0.1  
        self.gradient_clip = 1.0 
        self.pin_memory = True

        self.lr_scheduler_type = 'linear'  
        self.lr_scheduler_factor = 0.1
        self.patience = 5  
        self.min_delta = 0.001  
        self.warmup_steps = 100  
        self.scheduler_steps = 1000  
        self.max_grad_norm = 1.0  
        
        self.data_dir = "data/data"
        self.train_file = "data/train.txt"
        self.checkpoint_dir = "outputs/experiments"
        self.model_save_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        
        self.use_weighted_loss = True  
        self.use_balanced_sampler = False  
        self.use_augmentation = True  
        self.augment_minority_only = True  
        
        self.fusion_type = 'concat'
        self.interaction_dim = 256  
        self.use_layer_norm = True
        self.fusion_heads = 8  
        
        self.n_splits = 5  
        self.cv_random_seed = 42  
        self.stratified = True  
        self.shuffle_folds = True  
        
        self.use_amp = True  
        self.eval_frequency = 1  
        
        self.feature_adaptation = True
        self.unfreeze_layers = 2  
        self.adapter_dropout = 0.1
        