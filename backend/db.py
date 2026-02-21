import os
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path

# Load .env from backend folder
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "sia_plus")

if not MONGO_URI:
    raise ValueError("MONGO_URI not found. Check your .env file.")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

reviews_collection = db["reviews"]
