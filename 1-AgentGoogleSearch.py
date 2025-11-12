import os
import asyncio
import dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
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

root_agent = Agent(
    name="searching_agent",
    model = Gemini(
        model = "gemini-2.5-flash-lite",
        api_key = Google_API_key,
        retry_options = retry_config
    ),
    description = "A simple agent that answers questions",
    instruction = "You are a searching assistant. Use google assistant for current info or unsure",
    tools = [google_search],
)

async def main():
    runner = InMemoryRunner(agent=root_agent)
    response = await runner.run_debug("Whom do you think is going to win Big Boss 19 according to the recent trends?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
