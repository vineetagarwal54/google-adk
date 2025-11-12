import os
import dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.genai import types

# Load environment variables
dotenv.load_dotenv()
Google_API_key = os.getenv('GOOGLE_API_KEY')

retry_config = types.HttpRetryOptions(
    attempts = 5,
    exp_base = 7,
    initial_delay = 1,
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

# Agent definition for ADK Web UI
searching_agent = Agent(
    name="searching_agent",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        api_key = Google_API_key,
        retry_options = retry_config
    ),
    description = "A simple agent that answers questions using Google Search",
    instruction = "You are a searching assistant. Use google search for current information or when unsure. Provide accurate and helpful answers based on the search results.",
    tools = [google_search],
)
