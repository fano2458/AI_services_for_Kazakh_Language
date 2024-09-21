from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
# from models.text_summarization import

app = FastAPI()

class SummarizationRequest(BaseModel):
    text: str

class AskLLM(BaseModel):
    prompt: str
    question: str


@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    text = request.text

    # summary = model.summarize(text, max_length)
    summary = f"This is text summary of {text}"

    summary_json = {"summary": summary}

    return summary_json


@app.post("/llm")
async def ask_llm(request: AskLLM):
    prompt = request.prompt
    question = request.question

    return {}