import requests
import json
import logging

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF:latest"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def classify_sentiment(text: str) -> str:
    if not text or not text.strip():
        return "Neutral"

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": "Classify sentiment as Positive, Negative, or Neutral. Respond with one word."
                    },
                    {
                        "role": "user",
                        "content": text.strip()
                    }
                ],
                "options": {
                    "temperature": 0,
                    "num_predict": 5
                },
                "stream": False
            },
            timeout=30
        )

        lines = response.text.strip().split("\n")
        data = json.loads(lines[-1])
        sentiment = data["message"]["content"].strip()

        if sentiment.lower() in ["positive", "negative", "neutral"]:
            return sentiment.capitalize()
        return "Neutral"

    except Exception as e:
        logger.error(f"Sentiment classification error: {e}")
        return "Neutral"


def classify_issue(text: str) -> str:
    if not text or not text.strip():
        return "Other"

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Classify the issue into one category:\n"
                            "Delivery, Product, Support, Pricing, Other.\n"
                            "Respond with one word."
                        )
                    },
                    {
                        "role": "user",
                        "content": text.strip()
                    }
                ],
                "options": {
                    "temperature": 0,
                    "num_predict": 5
                },
                "stream": False
            },
            timeout=30
        )

        lines = response.text.strip().split("\n")
        data = json.loads(lines[-1])
        category = data["message"]["content"].strip()

        valid = ["Delivery", "Product", "Support", "Pricing", "Other"]
        if category.capitalize() in valid:
            return category.capitalize()
        return "Other"

    except Exception as e:
        logger.error(f"Issue classification error: {e}")
        return "Other"


def generate_answer(context, question):
    if not isinstance(context, str):
        if hasattr(context, '__iter__') and not isinstance(context, (str, bytes)):
            context = "\n".join(str(item) for item in context)
        else:
            context = str(context)

    stripped_context = context.strip()
    if not stripped_context:
        return "No relevant reviews found in the dataset."

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI customer experience analyst.\n\n"
                            "Your job:\n"
                            "1. Identify the main issues in the reviews.\n"
                            "2. Summarize the sentiment trends.\n"
                            "3. Suggest practical actions to fix the issues.\n\n"
                            "Keep answers concise and business-focused."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"REVIEWS:\n{stripped_context}\n\nQUESTION:\n{question}"
                    }
                ],
                "options": {
                    "temperature": 0.3,
                    "num_predict": 200
                },
                "stream": False
            },
            timeout=120
        )

        lines = response.text.strip().split("\n")
        data = json.loads(lines[-1])
        return data["message"]["content"].strip()

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return "Error generating response."
