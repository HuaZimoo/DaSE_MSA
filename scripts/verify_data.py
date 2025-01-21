import os
import pandas as pd
from PIL import Image
import chardet

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    return chardet.detect(raw_data)['encoding']

def verify_dataset(data_dir, annotation_file):
    """验证数据集完整性"""
    df = pd.read_csv(annotation_file)
    
    missing_files = []
    corrupted_files = []
    encoding_issues = []
    
    for guid in df['guid']:
        image_path = os.path.join(data_dir, f"{guid}.jpg")
        if not os.path.exists(image_path):
            missing_files.append(image_path)
        else:
            try:
                with Image.open(image_path) as img:
                    img.verify()  
            except Exception as e:
                corrupted_files.append((image_path, str(e)))
        
        text_path = os.path.join(data_dir, f"{guid}.txt")
        if not os.path.exists(text_path):
            missing_files.append(text_path)
        else:
            try:
                encoding = detect_encoding(text_path)
                if encoding is None:
                    encoding_issues.append(text_path)
                    continue
                    
                with open(text_path, 'r', encoding=encoding) as f:
                    text = f.read().strip()
                    if not text:  
                        corrupted_files.append((text_path, "Empty file"))
            except Exception as e:
                corrupted_files.append((text_path, str(e)))
    
    return missing_files, corrupted_files, encoding_issues

def main():
    data_dir = "data/data"
    train_file = "data/train.txt"
    test_file = "data/test_without_label.txt"
    
    print("验证训练集...")
    missing, corrupted, encoding_issues = verify_dataset(data_dir, train_file)
    if missing or corrupted or encoding_issues:
        print("训练集存在问题：")
        if missing:
            print(f"缺失文件：{missing}")
        if corrupted:
            print("\n损坏文件及原因：")
            for file, reason in corrupted:
                print(f"{file}: {reason}")
        if encoding_issues:
            print("\n编码问题文件：")
            print(encoding_issues)
    else:
        print("训练集完整性验证通过")
    
    print("\n验证测试集...")
    missing, corrupted, encoding_issues = verify_dataset(data_dir, test_file)
    if missing or corrupted or encoding_issues:
        print("测试集存在问题：")
        if missing:
            print(f"缺失文件：{missing}")
        if corrupted:
            print("\n损坏文件及原因：")
            for file, reason in corrupted:
                print(f"{file}: {reason}")
        if encoding_issues:
            print("\n编码问题文件：")
            print(encoding_issues)
    else:
        print("测试集完整性验证通过")

if __name__ == '__main__':
    main() 