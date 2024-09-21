from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional

# from models.text_summarization import
from models.irbis_llm import LLM


app = FastAPI()

# initialize llm once
llm = LLM()

class SummarizationRequest(BaseModel):
    text: str

class AskLLM(BaseModel):
    prompt: str


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

    answer = llm.ask_llm(prompt)

    return {"response": answer}