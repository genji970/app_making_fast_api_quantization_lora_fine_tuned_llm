# fastapi_lora_fine_tuned_llm
making chatbot using fine tuned llm in fast api 

## how to run ##
1) command line : uvicorn app:app --reload --port <your port number> in terminal

2) test after deployment
curl -X POST -H "Content-Type: application/json" -d '{"input_text": "What is AI?"}' http://<your_ip>/generate


## structure 1 ##
i) teck stack : FastApi , torch , llm from huggingface, peft , quantization 
ii) plan to add : gpu , sage maker, parallel processing
iii) models folder(model.py , Fine_tuning.py , Peft.py) , app.py , requirements.txt

## structure 2 ##

1) requirements.txt : packages need to install
2) Peft : contains lora_configuration hyperparameter
3) Dataset : json type data will be transformed to appropriate type(for llm 'role' : ... , 'content' : ...)
4) Fine_tuning : use this py to fine tune pretrained model before run app.py trainer.train() will make model to learn 

## methods to speed up llm response ##
1) data multiprocessing
2) batch : to prevent quality degrading, make groups according to sentence length

## rag ##

## fine tuning by sage maker ##

## use specific domain data ##

## result ##

not yet done : adding rag system , getting quialified dataset , using meta llama

