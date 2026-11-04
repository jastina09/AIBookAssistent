from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

import main

app = FastAPI(title="AI Book Assistant")

store = main.create_vector_store()
recommend_chain = main.make_recommend_chain(store)
explain_chain = main.make_explain_chain(store)
card_chain = main.make_card_chain(store)


class Question(BaseModel):
    text: str


@app.post("/recommend", response_class=PlainTextResponse)
def recommend(q: Question):
    return recommend_chain.invoke(q.text)


@app.post("/explain", response_class=PlainTextResponse)
def explain(q: Question):
    return explain_chain.invoke(q.text)


@app.post("/book-card", response_class=PlainTextResponse)
def book_card(q: Question):
    card = card_chain.invoke(q.text)
    return (
        f"Название: {card.title}\n"
        f"Автор: {card.author}\n"
        f"Жанр: {card.genre}\n"
        f"Кратко: {card.summary}\n"
        f"Кому понравится: {card.for_whom}"
    )