from fastapi import FastAPI
from pydantic import BaseModel

from models.irbis_llm import LLM
from models.ner import NER
from models.tts_generate import TTS


app = FastAPI()


class TextBasedRequest(BaseModel):
    """
    Request body for all endpoints. (for now)
    """
    text: str


@app.post("/summarize")
async def summarize(request: TextBasedRequest):
    """
    Summarizes a given text.
    To be implemented.
    """
    text = request.text

    # summary = model.summarize(text, max_length)
    summary = f"This is text summary of {text}"
    summary_json = {"text": summary}

    return summary_json


@app.post("/llm")
async def ask_llm(request: TextBasedRequest):
    """
    Queries a large language model (LLM) with a prompt and returns the answer.
    """
    llm = LLM()
    answer = llm.ask_llm(request.text)

    return {"text": ''.join(answer)}


@app.post("/ner")
async def ner(request: TextBasedRequest):
    """
    Performs Named Entity Recognition (NER) on a text and returns the results.
    """
    ner = NER()
    result = ner.predict(request.text)
    
    return {'text': str(result)}


@app.post("/tts")
async def tts(request: TextBasedRequest):
    """
    Converts text to speech (Text-To-Speech).
    """
    tts_model = TTS()
    result = tts_model.predict(request.text)

    return result
