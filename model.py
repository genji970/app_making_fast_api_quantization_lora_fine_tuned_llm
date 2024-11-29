from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import random


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LLMModel:
    def __init__(self , model_name , seed = 42):
        #seed 고정
        set_seed(seed)
        # 모델과 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt: str, max_length: int = 50) -> str:
        # 입력 텍스트를 토크나이저로 처리
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
        # 생성된 텍스트 디코딩
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)