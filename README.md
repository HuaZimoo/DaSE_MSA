# MSAClip: 基于CLIP的多模态情感分析

这是一个基于CLIP预训练模型的多模态情感分析项目实现，可以对配对的图像-文本数据进行情感标签（positive、neutral、negative）预测。

## 环境配置

本项目基于Python3实现，运行代码需要以下依赖：

- torch>=2.1.0
- transformers>=4.35.2
- numpy>=1.24.3
- pandas>=2.1.1
- scikit-learn>=1.3.2
- Pillow>=10.1.0
- chardet>=5.2.0
- tensorboard>=2.14.1
- tqdm>=4.66.1
- openai-clip>=1.0.0
- matplotlib>=3.8.2

可以直接运行以下命令安装：
```bash
pip install -r requirements.txt
```

## 项目结构
以下是主要文件的详细说明：
```
MSAClip/
├── README.md        # 项目说明文档
├── requirements.txt # 项目依赖
├── src/             # 源代码目录
│ ├── init.py
│ ├── data/          # 数据处理相关
│ │ ├── dataset.py   # 数据集实现
│ │ └── processor.py # 数据预处理
│ ├── models/        # 模型相关
│ │ ├── init.py
│ │ ├── sentiment_model.py # 情感分析模型
│ │ └── fusion.py          # 融合策略实现
│ └── utils/         # 工具函数
│ │ └── metrics.py   # 评估指标
└── scripts/         # 实验相关脚本文件
```

## 训练流程

1. 可以运行不同的实验策略：

运行模态消融实验
```
bash experiments/modality_ablation.sh
```

运行融合策略实验
```
bash experiments/fusion_strategy_exp.sh
```

运行数据处理策略实验
```
bash experiments/single_strategy_exp.sh
```

2. 也可以使用特定配置进行训练：
```
python experiments/train.py \
--fusion_type bilinear \
--use_balanced_sampler \
--use_augmentation \
--use_image --use_text
```

更多实验细节可以参考'experiments/'目录下的shell脚本。

## 预测流程

1. 训练完成后，可以对测试集进行预测：
```
python experiments/predict.py
```

这将生成与test_without_label.txt格式相同的预测结果。

## 引用

本代码部分基于以下开源项目：
- [CLIP](https://github.com/openai/CLIP)
- [Transformers](https://github.com/huggingface/transformers)
