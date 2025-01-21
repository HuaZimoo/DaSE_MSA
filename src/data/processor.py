import re
from typing import List, Optional

class TextPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.non_ascii_pattern = re.compile(r'[^\x00-\x7F]+')
        self.hashtag_pattern = re.compile(r'#(\w+)') 
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "]+", flags=re.UNICODE)
    
    def clean_text(self, text: str) -> str:
        text = self.url_pattern.sub('', text)
        text = self.non_ascii_pattern.sub('', text)
        text = self.hashtag_pattern.sub(r'\1', text)
        text = ' '.join(text.split())
        
        return text.strip()
    
    def normalize_text(self, text: str) -> str:
        """
        标准化文本
        """
        # 转换为小写
        text = text.lower()
        
        # 替换常见缩写
        text = text.replace("'m", " am")
        text = text.replace("'s", " is")
        text = text.replace("'re", " are")
        text = text.replace("'ll", " will")
        text = text.replace("'ve", " have")
        text = text.replace("'d", " would")
        text = text.replace("n't", " not")
        
        return text

class MultimodalPreprocessor:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, text, image):
        # 处理输入
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP 的默认最大长度
        )
        
        # 去掉批次维度
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        } 
    
    def test_text_preprocessing():
        preprocessor = TextPreprocessor()
        
        # 测试文本
        test_cases = [
            "grattis min griskulting!!!???? va bara tvungen oki s? sch ? @ingenkommeratttrodig #pig #happybday #wow #lovely #cut�� "
        ]
        
        # 处理并打印结果
        for text in test_cases:
            cleaned_text = preprocessor.clean_text(text)
            print(f"Original: {text}")
            print(f"Cleaned: {cleaned_text}\n")
