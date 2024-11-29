from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

class Peft_class():
    def configure(self):
        # LoRA 설정
        lora_config = LoraConfig(
            r=16,  # Low-rank 업데이트 행렬 차원
            lora_alpha=16,  # 스케일링 팩터
            lora_dropout=0.1,  # 드롭아웃 비율
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],  # QLoRA가 적용될 대상 모듈
        )

    #def get_hugging_token(self):
    #    huggingface-cli login

    def peft_model_build(self , base_model , lora_config):
        model = get_peft_model(base_model, lora_config)
        tokenizer = AutoTokenizer.from_pretrained(model)
        # meta llama 3.2 1B같이 따로 중간중간 token이 없는 경우
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model.resize_token_embeddings(len(tokenizer))
        return model , tokenizer