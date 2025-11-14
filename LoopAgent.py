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

dotenv.load_dotenv()
Google_API_key = os.getenv('GOOGLE_API_KEY')
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

#Initial Writer Agent Definition

intial_writer_agent = Agent(
    name="initial_writer_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=Google_API_key,
        retry_options=retry_config
    ),
    instruction="You are a helpful assistant that writes initial drafts of articles based on user prompts.",
    description="An agent that generates initial drafts of articles based on user prompts.",
    output_key="current_draft",
)

#critic Agent Definition
critic_agent = Agent(
    name="critic_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=Google_API_key,
        retry_options=retry_config
    ),
    instruction="You are a critical reviewer. Your job is to review the current draft: {current_draft} and provide constructive feedback and suggestions for improvement.",
    description="An agent that reviews drafts and provides constructive feedback.",
    output_key="critique",
)

#exitLoop Function Tool Definition
def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED' to exit the loop."""
    return {"status": "approved", "message": "The draft has been approved."}

#refine Agent Definition
refine_agent = Agent(
    name="refine_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=Google_API_key,
        retry_options=retry_config
    ),
    instruction="""You are a helpful assistant that refines the current draft: {current_draft} based on the critique: {critique}. Your task is to analyze the critique.
    - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function and nothing else.
    - OTHERWISE, rewrite the story draft to fully incorporate the feedback from the critique.",
    description="An agent that refines drafts based on critiques.""",
    output_key="current_draft",
    tools=[FunctionTool(exit_loop)]
)

#story_writer_agent Definition
story_writer_agent = LoopAgent(
    name="story_writer_agent",
    sub_agents=[critic_agent, refine_agent],
    max_iterations=3,
)

root_agent = SequentialAgent(
    name="StoryWritingCoordinatorAgent",
    sub_agents=[intial_writer_agent, story_writer_agent],   
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
        response = await runner.run_debug("Write a short story about a lighthouse keeper who discovers a mysterious, glowing map")
    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    # Extract and display just the final story
    print("\n" + "="*80)
    print("FINAL STORY:")
    print("="*80 + "\n")
    
    # Look for the current_draft in the state or the initial_writer_agent output
    if isinstance(response, list):
        # Find events from the initial_writer_agent (first agent that writes the story)
        story_text = None
        for event in response:
            if hasattr(event, 'author') and 'initial_writer' in event.author:
                if hasattr(event, 'content') and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            story_text = part.text
                            break
            # Also check the refine_agent for the refined version
            elif hasattr(event, 'author') and 'refine' in event.author:
                if hasattr(event, 'content') and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text and 'review' not in part.text.lower():
                            story_text = part.text
        
        if story_text:
            print(story_text)
            print("\n" + "="*80)
        else:
            # Fallback: print the last text event
            for event in reversed(response):
                if hasattr(event, 'content') and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            print(part.text)
                            print("\n" + "="*80)
                            return
    else:
        print(response)
        print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(main())
