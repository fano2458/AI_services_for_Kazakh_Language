from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional

# from models.text_summarization import
from models.irbis_llm import LLM
from models.ner import NER


app = FastAPI()

# initialize llm once
# llm = LLM()

class SummarizationRequest(BaseModel):
    text: str

class AskLLM(BaseModel):
    text: str

class FindNER(BaseModel):
    text: str


@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    text = request.text

    # summary = model.summarize(text, max_length)
    summary = f"This is text summary of {text}"

    summary_json = {"text": summary}

    return summary_json


@app.post("/llm")
async def ask_llm(request: AskLLM):
    prompt = request.text
    llm = LLM()
    answer = llm.ask_llm(prompt)
    answer = ''.join(answer)
    # answer = "response"

    return {"text": answer}


@app.post("/ner")
async def ner(request: FindNER):
    text = request.text
    ner = NER()

    result = ner.predict(text)
    return result
