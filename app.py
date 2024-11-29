from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from models.model import LLMModel

#fastapi 애플리케이션 실행 코드 : uvicorn app:app --reload --port 8000
# 클라이언트 실행 : python code 실행 , python client.py

app = FastAPI()

# LLM 모델 초기화
model_name="C:/Users/fine_tuned_model" # fine tuning한 경우
model_name = "gpt2"
llm = LLMModel(model_name = model_name)

# 요청 데이터 구조 정의
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100

@app.get("/") # it is decorator
def read_root():
    return {"message": "Welcome to the LLM API!"}

@app.post("/generate")
def generate_text(request: GenerateRequest):
    try:
        # LLM으로 텍스트 생성
        response = llm.generate(request.prompt, max_length=request.max_length)
        return {"generated_text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))