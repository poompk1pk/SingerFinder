import asyncio
import logging

import fastapi
from fastapi import FastAPI
from fastapi import FastAPI, Request
from singer_finder_service.singer_finder import SingerFinderService

from sklearn.feature_extraction.text import TfidfVectorizer
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from fastapi.templating import Jinja2Templates
import threading
import time

from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/statics", StaticFiles(directory="statics"), name="statics")

service = None

@app.on_event("startup")
async def startup_event():
    print('Starting SFS...!')
    global service
    service = SingerFinderService()
    print('SFS is ready!')




templates = Jinja2Templates(directory="templates")

@app.get("/")
async def search(request: Request,query: str = ''):
    start = time.perf_counter()

    if query == '':
        return templates.TemplateResponse("search.html", {"request": request,"query": query, "result": []})
    result = await service.search(query)

    usedTime = time.perf_counter() - start
    print(usedTime)
    return templates.TemplateResponse("search.html", {"request": request,"query": query, "result": result,'time': usedTime})
