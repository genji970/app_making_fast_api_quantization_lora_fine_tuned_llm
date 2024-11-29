# fastapi_lora_fine_tuned_llm
making chatbot using fine tuned llm in fast api 

## how to run ##
command line : uvicorn app:app --reload --port <your port number> in terminal

## structure 1 ##
teck stack : FastApi , torch , llm from huggingface, peft , quantization 
plan to add : gpu , sage maker, parallel processing

## structure 2 ##

1) requirements.txt : packages need to install
2) Peft : contains lora_configuration hyperparameter
3) Dataset : json type data will be transformed to appropriate type(for llm 'role' : ... , 'content' : ...)
4) Fine_tuning : use this py to fine tune pretrained model before run app.py trainer.train() will make model to learn 



not yet done : adding rag system , getting quialified dataset , using meta llama

