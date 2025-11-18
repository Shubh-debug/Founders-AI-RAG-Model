"""
Intelligent startup research agent using LangChain tools and OpenAI models.

Provides automated startup knowledge retrieval with metric extraction, sector
classification, and enhanced response generation with fallback mechanisms for reliability.

This agent helps founders and operators ask conversational questions about startup
strategies, metrics, and case studies. It integrates with the Founders AI knowledge
base to deliver data-driven startup insights.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available, using fallback implementation")
    LANGCHAIN_AVAILABLE = False


class FounderQueryInput(BaseModel):
    """Input for founder research queries"""
    query: str = Field(..., description="The founder or startup research query")
    context: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None


class FounderResearchOutput(BaseModel):
    """Output for founder research"""
    response: str = Field(..., description="The startup research response with actionable insights")
    citations: List[str] = Field(default_factory=list, description="Extracted startup metrics and citations")
    sector: str = Field(..., description="Business sector classification (Fintech, Crypto, AI, E-commerce, Social, Other)")
    confidence: float = Field(..., description="Confidence score of the response (0.0-1.0)")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source case studies used")
    tools_used: List[str] = Field(default_factory=list, description="Tools used by the agent")


if LANGCHAIN_AVAILABLE:

    class StartupMetricsTool(BaseTool):
        """Tool for extracting startup metrics from text"""
        name: str = "extract_startup_metrics"
        description: str = "Extract business metrics from startup text. Identifies ARR, MRR, MAU, DAU, GMV, funding rounds, and growth percentages."

        def _run(self, text: str) -> str:
            """Extract startup metrics from text"""
            try:
                from .startup_tools import extract_startup_citations
                metrics = extract_startup_citations(text)
                if metrics:
                    return f"Found startup metrics: {', '.join(metrics)}"
                else:
                    return "No key metrics found in the text."
            except Exception as e:
                logger.error(f"Metric extraction failed: {e}")
                return f"Error extracting metrics: {str(e)}"

        async def _arun(self, text: str) -> str:
            """Async version of metric extraction"""
            return self._run(text)


    class SectorClassificationTool(BaseTool):
        """Tool for classifying startup sector and business model"""
        name: str = "classify_startup_sector"
        description: str = "Classify startup by sector: Fintech, Crypto, AI, E-commerce, Social, or Other."

        def _run(self, text: str) -> str:
            """Classify startup sector"""
            try:
                from .startup_classifier import startup_classifier
                result = startup_classifier.classify(text)
                return f"Sector: {result['category']} (confidence: {result['confidence']:.2f})"
            except Exception as e:
                logger.error(f"Sector classification failed: {e}")
                return f"Error in classification: {str(e)}"

        async def _arun(self, text: str) -> str:
            """Async version of sector classification"""
            return self._run(text)


    class StartupResearchTool(BaseTool):
        """Tool for searching startup case studies and retrieving insights"""
        name: str = "search_startup_case_studies"
        description: str = "Search startup case studies and retrieve relevant business insights, metrics, and strategy examples."

        def _run(self, query: str) -> str:
            """Search startup case studies (synchronous wrapper)"""
            try:
                return f"Startup research query: {query}. Please use async version for full functionality."
            except Exception as e:
                logger.error(f"Startup research failed: {e}")
                return f"Error in startup research: {str(e)}"

        async def _arun(self, query: str) -> str:
            """Async version of startup research"""
            try:
                from .lightweight_llm_rag_founders_ai import lightweight_llm_rag
                result = await lightweight_llm_rag.process_query(query=query, top_k=5)
                return f"Startup case study insights: {result.get('response', 'No results found')}"
            except Exception as e:
                logger.error(f"Startup research failed: {e}")
                return f"Error in startup research: {str(e)}"


class LangChainStartupAgent:
    """
    LangChain-based startup research agent with tool integration and fallback support.

    This agent helps founders and operators ask conversational questions about startup
    strategies, metrics, and case studies. It uses LangChain tools to extract metrics,
    classify sectors, and search the case study knowledge base.
    """

    def __init__(self):
        self.llm = None
        self.tools = []
        self.agent_executor = None
        self.initialized = False

    async def initialize(self):
        """Initialize the LangChain startup research agent"""
        if self.initialized:
            return

        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, using fallback implementation")
            self.initialized = True
            return

        try:
            # Initialize OpenAI LLM with startup-focused parameters
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.3,
            )

            # Set up startup-focused tools
            self.tools = [
                StartupMetricsTool(),
                SectorClassificationTool(),
                StartupResearchTool()
            ]

            # Create agent prompt for startup research
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a startup research advisor helping founders and operators.

Your role:
1. Answer startup and business strategy questions based on real case studies
2. Use tools to extract metrics and classify business models
3. Provide data-driven insights from successful companies
4. Focus on actionable advice for founders

Response guidelines:
- Lead with clear, direct answers to the founder's question
- Include specific examples from the case studies
- Cite metrics and data points when relevant
- Use bullet points for multiple insights
- Explain business strategies and tactics with real examples
- Keep advice practical and actionable

Available tools:
- search_startup_case_studies: Find case study insights
- extract_startup_metrics: Extract ARR, MAU, GMV, funding data
- classify_startup_sector: Identify business sector (Fintech, Crypto, AI, etc)"""),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            # Create agent executor
            agent = create_openai_tools_agent(self.llm, self.tools, prompt)
            # Cast agent to Any to satisfy AgentExecutor typing (create_openai_tools_agent may return a Runnable)
            from typing import cast
            self.agent_executor = AgentExecutor(
                agent=cast(Any, agent),
                tools=self.tools,
                verbose=True
            )

            self.initialized = True
            logger.info("LangChain Startup Research Agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LangChain Startup Research Agent: {e}")
            raise

    async def research(
        self,
        query: str,
        session_id: Optional[str] = None
    ) -> FounderResearchOutput:
        """
        Perform startup research using the LangChain agent.

        This method processes the founder's query through the agent pipeline,
        extracting metrics, classifying sectors, and synthesizing insights
        from case studies.

        Args:
            query: The founder's research question
            session_id: Optional session identifier for tracking

        Returns:
            FounderResearchOutput with response, citations, and metadata
        """
        if not self.initialized:
            await self.initialize()

        if not LANGCHAIN_AVAILABLE or not self.agent_executor:
            return await self._fallback_research(query, session_id)

        try:
            # Run agent
            result = self.agent_executor.invoke({"input": query})
            agent_response = result["output"]

            # Get case study sources
            from .lightweight_llm_rag_founders_ai import lightweight_llm_rag
            rag_result = await lightweight_llm_rag.process_query(
                query=query,
                top_k=5
            )
            sources = rag_result.get("sources", [])

            # Extract metrics
            from .startup_tools import extract_startup_citations
            citations = extract_startup_citations(agent_response)

            # Classify sector
            from .startup_classifier import startup_classifier
            sector_result = startup_classifier.classify(query)
            sector = sector_result["category"]
            confidence = sector_result["confidence"]

            # Build output
            output = FounderResearchOutput(
                response=agent_response,
                citations=citations,
                sector=sector,
                confidence=confidence,
                sources=sources,
                tools_used=[
                    "search_startup_case_studies",
                    "extract_startup_metrics",
                    "classify_startup_sector"
                ]
            )

            return output

        except Exception as e:
            logger.error(f"LangChain agent research failed: {e}")
            return await self._fallback_research(query, session_id)

    async def _fallback_research(
        self,
        query: str,
        session_id: Optional[str] = None
    ) -> FounderResearchOutput:
        """
        Fallback research implementation when LangChain is not available.

        Uses lightweight RAG directly to retrieve case study insights
        without the agent framework.
        """
        try:
            # Query RAG directly
            from .lightweight_llm_rag_founders_ai import lightweight_llm_rag
            rag_result = await lightweight_llm_rag.process_query(
                query=query,
                top_k=5
            )
            response = rag_result.get("response", "")
            sources = rag_result.get("sources", [])

            # Extract metrics
            from .startup_tools import extract_startup_citations
            citations = extract_startup_citations(response)

            # Classify sector
            from .startup_classifier import startup_classifier
            sector_result = startup_classifier.classify(query)
            sector = sector_result["category"]
            confidence = sector_result["confidence"]

            # Enhance response
            enhanced_response = await self._enhance_response(
                query, response, sector, citations
            )

            output = FounderResearchOutput(
                response=enhanced_response,
                citations=citations,
                sector=sector,
                confidence=confidence,
                sources=sources,
                tools_used=["fallback_research"]
            )

            return output

        except Exception as e:
            logger.error(f"Fallback research failed: {e}")
            return FounderResearchOutput(
                response=f"Error in startup research: {str(e)}",
                citations=[],
                sector="Other",
                confidence=0.0,
                sources=[],
                tools_used=[]
            )

    async def _enhance_response(
        self,
        query: str,
        response: str,
        sector: str,
        citations: List[str]
    ) -> str:
        """
        Enhance the response with sector context and structured citations.

        Adds business context and formats metrics for clarity.
        """
        try:
            sector_context = {
                "Fintech": "This relates to financial technology and digital payments.",
                "Crypto": "This relates to cryptocurrency and blockchain businesses.",
                "AI": "This relates to artificial intelligence and machine learning applications.",
                "E-commerce": "This relates to online retail and marketplace businesses.",
                "Social": "This relates to social media and community platforms.",
                "Other": "This relates to general startup strategy and operations."
            }

            # Add sector context if available
            enhanced = f"{sector_context.get(sector, '')}\n\n{response}"

            # Add extracted metrics
            if citations:
                enhanced += f"\n\nKey Metrics: {', '.join(citations)}"

            return enhanced

        except Exception as e:
            logger.error(f"Response enhancement failed: {e}")
            return response


# Global instance for easy importing
langchain_startup_agent = LangChainStartupAgent()
# Backward-compatible alias expected by lifecycle initializer
langchain_legal_agent = langchain_startup_agent