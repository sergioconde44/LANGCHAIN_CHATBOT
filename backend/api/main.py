from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend.agent.pipeline import rag_pipeline

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    answer = rag_pipeline.ask(request.message)
    return ChatResponse(response=answer)

@app.post("/reload-data")
async def reload_data():
    rag_pipeline.reload_data()
    return {"status": "reloaded"}
