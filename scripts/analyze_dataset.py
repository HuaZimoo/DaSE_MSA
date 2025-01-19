import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib as mpl

def set_matplotlib_chinese():
    """设置matplotlib中文字体"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac系统
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def analyze_dataset(train_file, data_dir):
    """分析数据集的基本信息"""
    # 设置中文字体
    set_matplotlib_chinese()
    
    # 直接读取文本文件
    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 解析数据
    data = []
    for line in lines[1:]:  # 跳过标题行
        if line.strip():  # 跳过空行
            try:
                guid, sentiment = line.strip().split(',')
                # 将文本标签转换为数字标签
                label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
                label = label_map[sentiment]
                data.append({'guid': guid, 'label': label})
            except Exception as e:
                print(f"解析行时出错: {line.strip()}, 错误: {e}")
                continue
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 打印原始数据的前几行用于调试
    print("数据预览:")
    print(df.head())
    print(f"\n数据形状: {df.shape}")
    
    # 标签映射说明
    print("\n标签映射:")
    print("0: negative (负面)")
    print("1: neutral (中性)")
    print("2: positive (正面)")
    
    # 统计标签分布
    label_dist = Counter(df['label'])
    
    # 分析文本长度
    text_lengths = []
    image_sizes = []
    
    for guid in df['guid']:
        try:
            # 读取文本，尝试不同的编码方式
            text_path = os.path.join(data_dir, f"{guid}.txt")
            text = None
            for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']:
                try:
                    with open(text_path, 'r', encoding=encoding) as f:
                        text = f.read().strip()
                        break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if text is None:
                print(f"无法解码文本文件 {guid}.txt")
                continue
                
            text_lengths.append(len(text))
            
            # 读取图片尺寸
            image_path = os.path.join(data_dir, f"{guid}.jpg")
            with Image.open(image_path) as img:
                image_sizes.append(img.size)
        except Exception as e:
            print(f"处理样本 {guid} 时出错: {e}")
            continue
    
    # 打印数据集统计信息
    print("\n=== 数据集统计信息 ===")
    print(f"总样本数: {len(df)}")
    print("\n标签分布:")
    for label, count in sorted(label_dist.items()):
        print(f"标签 {label}: {count} 样本 ({count/len(df)*100:.2f}%)")
    
    if text_lengths:
        # 文本长度统计
        text_lengths = pd.Series(text_lengths)
        print("\n文本长度统计:")
        print(f"最短文本: {text_lengths.min()} 字符")
        print(f"最长文本: {text_lengths.max()} 字符")
        print(f"平均长度: {text_lengths.mean():.2f} 字符")
        print(f"中位长度: {text_lengths.median()} 字符")
    
    if image_sizes:
        # 图像尺寸统计
        widths, heights = zip(*image_sizes)
        print("\n图像尺寸统计:")
        print(f"宽度范围: {min(widths)} - {max(widths)} 像素")
        print(f"高度范围: {min(heights)} - {max(heights)} 像素")
        
        # 绘制可视化图表
        plt.figure(figsize=(15, 5))
        
        # 标签分布饼图
        plt.subplot(131)
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        labels = ['负面', '中性', '正面']
        plt.pie(label_dist.values(), labels=labels,
                autopct='%1.1f%%', colors=colors)
        plt.title('情感标签分布')
        
        # 文本长度分布直方图
        plt.subplot(132)
        plt.hist(text_lengths, bins=30, color='#66b3ff')
        plt.title('文本长度分布')
        plt.xlabel('字符数')
        plt.ylabel('样本数')
        
        # 图像尺寸散点图
        plt.subplot(133)
        plt.scatter(widths, heights, alpha=0.5, color='#99ff99')
        plt.title('图像尺寸分布')
        plt.xlabel('宽度(像素)')
        plt.ylabel('高度(像素)')
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    # 设置文件路径
    train_file = "data/train.txt"
    data_dir = "data/data"
    
    # 分析数据集
    analyze_dataset(train_file, data_dir) 