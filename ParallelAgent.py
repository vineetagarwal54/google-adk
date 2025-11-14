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

dotenv.load_dotenv()
Google_API_key = os.getenv('GOOGLE_API_KEY')
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)  # Retry on these HTTP errors

#Tech Research Agent Definition
tech_research_agent = Agent(
    name="TechResearchAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=Google_API_key,
        retry_options=retry_config
    ),
    instruction="""You are a specialized tech research agent. Your only job is to use the google_search tool to find 2-3 pieces of relevant information on the given technology topic and present the findings with citations.""",
    description="An agent that conducts research on a given technology topic using Google Search and provides summarized findings with citations.",
    tools=[google_search],
    output_key="tech_research_findings"
)

#Medical Research Agent Definition
medical_research_agent = Agent(
    name="MedicalResearchAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=Google_API_key,
        retry_options=retry_config
    ),
    instruction="""You are a specialized medical research agent. Your only job is to use the google_search tool to find 2-3 pieces of relevant information on the given medical topic and present the findings with citations.""",
    description="An agent that conducts research on a given medical topic using Google Search and provides summarized findings with citations.",
    tools=[google_search],
    output_key="medical_research_findings"
)

financial_research_agent = Agent(
    name="FinancialResearchAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        api_key=Google_API_key,
        retry_options=retry_config
    ),
    instruction="""You are a specialized financial research agent. Your only job is to use the google_search tool to find 2-3 pieces of relevant information on the given financial topic and present the findings with citations.""",
    description="An agent that conducts research on a given financial topic using Google Search and provides summarized findings with citations.",
    tools=[google_search],
    output_key="financial_research_findings"
)

aggregator_agent = Agent(
    name="AggregatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # It uses placeholders to inject the outputs from the parallel agents, which are now in the session state.
    instruction="""Combine these three research findings into a single executive summary:

    **Technology Trends:**
    {tech_research_findings}
    
    **Health Breakthroughs:**
    {medical_research_findings}
    
    **Finance Innovations:**
    {financial_research_findings}
    
    Your summary should highlight common themes, surprising connections, and the most important key takeaways from all three reports. The final summary should be around 200 words.""",
    output_key="executive_summary",  # This will be the final output of the entire system.
)

parallel_research_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[tech_research_agent, medical_research_agent, financial_research_agent],
)

# This SequentialAgent defines the high-level workflow: run the parallel team first, then run the aggregator.
root_agent = SequentialAgent(
    name="ResearchSystem",
    sub_agents=[parallel_research_team, aggregator_agent],
)

async def main():
    runner = InMemoryRunner(agent=root_agent)
    response = await runner.run_debug("Latest advancements in technology, health, and finance")     
    print("Executive Summary:\n")
    print(response)