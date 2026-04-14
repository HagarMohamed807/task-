"""
Configuration — LLM API settings (loaded from environment / .env).
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file if present

GEMINI_API_KEY  = os.getenv("GROQ_API_KEY", "")   # set in .env or environment variable
MODEL_NAME      = "llama-3.3-70b-versatile"          # Free Groq model. Alt: "llama3-8b-8192" (faster/lighter)
GEMINI_BASE_URL = "https://api.groq.com/openai/v1/"  # Groq OpenAI-compatible endpoint
