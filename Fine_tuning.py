from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

#custome lib import
import Dataset
from Peft import Peft_class

# 데이터 로드
data_name = "tatsu-lab/alpaca"
dataloader = Dataset.Data_Load()
pre_train_data , pre_test_data = dataloader.data_load(data_name)

null_processing = Dataset.Data_Load()
train_data , test_data = null_processing.null_process(pre_train_data , pre_test_data)

# 모델과 토크나이저 로드
model_name = "gpt2"

base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  device_map="auto",             # GPU와 CPU를 자동 분배
                                                  torch_dtype="auto",            # 자동으로 적절한 데이터 타입(FP32, FP16 등) 선택
                                                  offload_folder="./offload",    # 메모리가 부족할 경우 CPU로 데이터를 오프로드
                                                  offload_state_dict=True)        # 가중치도 필요 시 CPU로 오프로드

#기존 model freeze
for param in base_model.parameters():
    param.requires_grad = False

# peft lora processing
peft_class = Peft_class()
lora_config = peft_class.configure()
model , tokenizer = peft_class.peft_model_build(base_model , lora_config)

# 데이터 전처리
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

tokenized_data = train_data.map(preprocess_function, batched=True)

# TrainingArguments 정의
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    seed=42,  # Seed 고정
    logging_dir='./logs',
    logging_steps=500,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)

# 모델 Fine-tuning
trainer.train()

# Fine-tuned 모델 저장
trainer.save_model("C:/Users/fine_tuned_model")