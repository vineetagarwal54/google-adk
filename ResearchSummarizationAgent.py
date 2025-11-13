import os
import asyncio
import warnings
import logging
import dotenv
from google.adk.agents import Agent, ParallelAgent, SequentialAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.getLogger('google').setLevel(logging.ERROR)

# Load environment variables
dotenv.load_dotenv()    
Google_API_key = os.getenv('GOOGLE_API_KEY')
retry_config = types.HttpRetryOptions(
    attempts = 5,
    exp_base = 7,
    initial_delay = 1,
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)
# Research Agent Definition

research_agent = Agent(
    name="ResearchAgent",
    model= Gemini(
        model="gemini-2.5-flash-lite",
        api_key=Google_API_key,
        retry_options=retry_config  
    ),
    instruction= """You are a specialized research agent. Your only job is to use the google_search tool to find 2-3 pieces of relevant information on the given topic and present the findings with citations.""",
    description="An agent that conducts research on a given topic using Google Search and provides summarized findings with citations.",
    tools=[google_search],
    output_key="research_findings"
)

#Summarization Agent Definition

summarizer_agent = Agent(
    name  ="SummarizerAgent",
    model= Gemini(
        model="gemini-2.5-flash-lite",
        api_key=Google_API_key,
        retry_options=retry_config  
    ),
    instruction="""Read the provided research findings: {research_findings} Create a concise summary as a bulleted list with 3-5 key points.""",
    description="An agent that summarizes research findings into concise bullet points.",
    output_key="summary",
)

root_agent = Agent(
    name="ResearchCoordinatorAgent",
    model= Gemini(
        model="gemini-2.5-flash-lite",
        api_key=Google_API_key,
        retry_options=retry_config  
    ),
   instruction="""You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.
1. First, you MUST call the `ResearchAgent` tool to find relevant information on the topic provided by the user.
2. Next, after receiving the research findings, you MUST call the `SummarizerAgent` tool to create a concise summary.
3. Finally, present the final summary clearly to the user as your response.
4. The Summary should be atleast 200 words""",
    description="An agent that coordinates research and summarization tasks.",
    tools=[AgentTool(research_agent), AgentTool(summarizer_agent)],
)

async def main():
    # Suppress stdout and stderr during debug run for clean output
    import sys
    from io import StringIO
    
    runner = InMemoryRunner(agent=root_agent)
    
    # Capture all debug output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        # Get the response
        response = await runner.run_debug("Why was the aws cloud service outage in november 2025?")
    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    # Extract and display just the final text response
    print("\n" + "="*80)
    print("FINAL RESPONSE:")
    print("="*80)
    
    # Get the last event with text content from the agent
    if isinstance(response, list):
        for event in reversed(response):
            if hasattr(event, 'content') and event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(part.text)
                        print("="*80)
                        return
    else:
        print(response)
        print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
