from fastapi import FastAPI, BackgroundTasks
import pandas as pd
import csv
import numpy as np
from datetime import datetime, timedelta
from pydantic import BaseModel

from db import reviews_collection
from embeddings import get_embedding
from vector_store import VectorStore
from chat import generate_answer, classify_sentiment, classify_issue

app = FastAPI()
vector_store = VectorStore()


# -------------------------------
# CSV LOADER
# -------------------------------
def load_data_from_csv():
    df = pd.read_csv("../data/reviews.csv")
    reviews_collection.delete_many({})

    records = []
    for _, row in df.iterrows():
        sentiment = classify_sentiment(row["text"])
        category = classify_issue(row["text"])
        emb = get_embedding(row["text"])

        records.append({
            "text": row["text"],
            "timestamp": row["timestamp"],
            "sentiment": sentiment,
            "category": category,
            "embedding": emb
        })

    if records:
        reviews_collection.insert_many(records)

    vector_store.rebuild_index()
    print(f"Loaded {len(records)} reviews from CSV.")


@app.post("/load-csv")
def load_csv():
    load_data_from_csv()
    return {"message": "CSV reloaded successfully"}


# -------------------------------
# STARTUP
# -------------------------------
@app.on_event("startup")
def startup():
    count = reviews_collection.count_documents({})

    if count == 0:
        load_data_from_csv()
    else:
        vector_store.rebuild_index()


# -------------------------------
# ALERT CHECK
# -------------------------------
def check_alerts():
    one_min_ago = datetime.now() - timedelta(minutes=1)

    count = reviews_collection.count_documents({
        "sentiment": "Negative",
        "timestamp": {"$gte": one_min_ago.strftime("%Y-%m-%d %H:%M:%S")}
    })

    if count >= 3:
        print("🚨 ALERT: Negative review spike detected!")


# -------------------------------
# BACKGROUND PROCESSOR
# -------------------------------
def process_review_background(text, timestamp):
    sentiment = classify_sentiment(text)
    category = classify_issue(text)
    emb = get_embedding(text)

    record = {
        "text": text,
        "timestamp": timestamp,
        "sentiment": sentiment,
        "category": category,
        "embedding": emb
    }

    reviews_collection.insert_one(record)

    vector_store.index.add(
        np.array([emb]).astype("float32")
    )
    vector_store.texts.append(text)

    check_alerts()


# -------------------------------
# MODELS
# -------------------------------
class ReviewInput(BaseModel):
    text: str


# -------------------------------
# BASIC ENDPOINTS
# -------------------------------
@app.get("/")
def home():
    return {"status": "SIA+ backend running"}


@app.get("/reviews")
def get_reviews():
    data = list(reviews_collection.find({}, {"_id": 0}))
    return {"reviews": data}


@app.get("/search")
def search_reviews(query: str):
    query_emb = get_embedding(query)
    results = vector_store.search(query_emb)
    return {"results": results}


@app.get("/chat")
def chat(query: str):
    query_emb = get_embedding(query)
    results = vector_store.search(query_emb, top_k=5)

    if not results:
        return {"context": [], "answer": "No relevant reviews found."}

    context_text = "\n".join(results)
    answer = generate_answer(context_text, query)

    return {"context": results, "answer": answer}


# -------------------------------
# EXECUTIVE SUMMARY ENDPOINT
# -------------------------------
@app.get("/summary")
def generate_summary():
    try:
        last_24h = datetime.now() - timedelta(hours=24)

        recent_reviews = list(
            reviews_collection.find(
                {"timestamp": {"$gte": last_24h.strftime("%Y-%m-%d %H:%M:%S")}},
                {"_id": 0}
            )
        )

        if not recent_reviews:
            return {"summary": "No reviews in the last 24 hours."}

        texts = [r["text"] for r in recent_reviews]
        context = "\n".join(texts)

        summary_prompt = (
            "You are an AI business analyst.\n\n"
            "Analyze the following customer reviews from the last 24 hours.\n"
            "Provide:\n"
            "1. Overall sentiment trend\n"
            "2. Main recurring issues\n"
            "3. Positive highlights\n"
            "4. Recommended actions\n\n"
            "Keep it concise and executive-level."
        )

        summary = generate_answer(context, summary_prompt)

        return {"summary": summary}

    except Exception as e:
        return {"summary": f"Error generating summary: {e}"}


# -------------------------------
# LIVE INGESTION
# -------------------------------
@app.post("/add-review-to-csv")
def add_review_to_csv(review: ReviewInput, background_tasks: BackgroundTasks):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open("../data/reviews.csv", "a", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([review.text, timestamp])

    background_tasks.add_task(
        process_review_background,
        review.text,
        timestamp
    )

    return {"message": "Review added. Processing in background."}
