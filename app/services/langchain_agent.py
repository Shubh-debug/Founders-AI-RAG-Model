"""
Founders AI - Startup Research Agent (LangChain-friendly with safe fallback)

Behaviour:
- Prefer LangChain agent APIs if available:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
- If LangChain agent APIs are missing (different langchain versions), use a safe fallback
  that calls local tools and an LLM wrapper to synthesize responses.
- Minimal, clear documentation. Async-friendly.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import HumanMessage, AIMessage, SystemMessage
    from langchain_core.messages import BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available, using fallback implementation")
    LANGCHAIN_AVAILABLE = False


class FounderQueryInput(BaseModel):
    query: str
    context: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None


class FounderResearchOutput(BaseModel):
    response: str
    citations: List[str] = Field(default_factory=list)
    sector: str = "Other"
    confidence: float = 0.0
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)


# -----------------------
# Try to import LangChain Agent symbols; create fallback shims if missing
# -----------------------
try:
    # Preferred imports (user requested)
    from langchain.agents import AgentExecutor, create_openai_tools_agent  # type: ignore
    LANGCHAIN_AGENT_AVAILABLE = True
    logger.info("LangChain agent APIs available.")
except Exception:
    AgentExecutor = None
    create_openai_tools_agent = None
    LANGCHAIN_AGENT_AVAILABLE = False
    logger.info("LangChain agent APIs not available — using fallback agent implementation.")


# -----------------------
# LLM wrapper (langchain_openai or openai fallback)
# -----------------------
class LLMClient:
    """Simple LLM client wrapper. Uses langchain_openai.ChatOpenAI if installed, else openai SDK."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self._client_type = None
        self._langchain_client = None
        self._openai = None
        self._init()

    def _init(self):
        # Try langchain_openai first
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            self._langchain_client = ChatOpenAI(model=self.model, temperature=self.temperature)
            self._client_type = "langchain_openai"
            logger.info("LLMClient: using langchain_openai.ChatOpenAI")
            return
        except Exception:
            pass

        # Fallback to openai
        try:
            import openai  # type: ignore
            self._openai = openai
            self._client_type = "openai"
            logger.info("LLMClient: using openai SDK")
        except Exception:
            self._client_type = None
            logger.warning("LLMClient: no LLM backend available (install openai or langchain_openai).")

    async def generate(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        """
        messages: [{"role":"system|user|assistant", "content":"..."}]
        returns: assistant content string
        """
        if self._client_type == "langchain_openai" and self._langchain_client is not None:
            try:
                from langchain_core.messages import (
                    SystemMessage, HumanMessage, AIMessage, BaseMessage
                )
                from typing import cast, List

                # Convert to LC messages
                def to_lc(m):
                    if m["role"] == "system":
                        return SystemMessage(content=m["content"])
                    elif m["role"] == "assistant":
                        return AIMessage(content=m["content"])
                    return HumanMessage(content=m["content"])

                lc_messages = [to_lc(m) for m in messages]

                # ---- FIX: cast to satisfy type checker ----
                lc_messages_casted: List[BaseMessage] = cast(List[BaseMessage], lc_messages)

                # Preferred: ainvoke
                if hasattr(self._langchain_client, "ainvoke"):
                    resp = await self._langchain_client.ainvoke(lc_messages_casted)
                    content = getattr(resp, "content", None)
                    return content if isinstance(content, str) else str(content)

                # Fallback: agenerate (must be batched)
                if hasattr(self._langchain_client, "agenerate"):
                    resp = await self._langchain_client.agenerate(
                        messages=[lc_messages_casted]
                    )
                    gens = getattr(resp, "generations", None)
                    if gens and gens[0] and gens[0][0]:
                        return gens[0][0].text
                    return str(resp)

            except Exception as e:
                logger.warning(f"langchain_openai generate failed: {e}")




        if self._client_type == "openai" and self._openai is not None:
            def _sync_call():
                try:
                    from openai import OpenAI
                    client = OpenAI()

                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in messages
                        ],
                        max_tokens=max_tokens,
                        temperature=self.temperature,
                    )

                    return response.choices[0].message.content

                except Exception as e:
                    logger.error(f"OpenAI SDK ChatCompletion error: {e}")
                    return "LLM error"

            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, _sync_call)
            try:
                if isinstance(resp, dict) and resp.get("choices"):
                    choice = resp["choices"][0]
                    text = choice.get("message", {}).get("content") or choice.get("text")
                    if text:
                        return text
                # some SDKs return objects with attributes
                try:
                    return resp.choices[0].message.content
                except Exception:
                    return str(resp)
            except Exception as e:
                logger.warning(f"openai parse failed: {e}")
                return str(resp)

        raise RuntimeError("No LLM backend available. Install openai or langchain_openai.")


# -----------------------
# Tools (wrappers that call your local code)
# -----------------------
def extract_startup_metrics(text: str) -> List[str]:
    """Return list of metric strings. Uses local extractor if available."""
    try:
        from .startup_tools import extract_startup_citations
        metrics = extract_startup_citations(text)
        return metrics or []
    except Exception as e:
        logger.debug(f"extract_startup_metrics fallback: {e}")
        return []


def classify_startup_sector(text: str) -> Dict[str, Any]:
    """Return {category, confidence}. Uses local classifier if available."""
    try:
        from .startup_classifier import startup_classifier
        res = startup_classifier.classify(text)
        return {"category": res.get("category", "Other"), "confidence": float(res.get("confidence", 0.0))}
    except Exception as e:
        logger.debug(f"classify_startup_sector fallback: {e}")
        t = text.lower()
        if "fintech" in t or "payment" in t:
            return {"category": "Fintech", "confidence": 0.6}
        if "crypto" in t or "blockchain" in t:
            return {"category": "Crypto", "confidence": 0.6}
        if "ai" in t or "machine learning" in t:
            return {"category": "AI", "confidence": 0.6}
        if "ecom" in t or "marketplace" in t:
            return {"category": "E-commerce", "confidence": 0.6}
        return {"category": "Other", "confidence": 0.4}


async def search_startup_case_studies(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Call your lightweight RAG module if available."""
    try:
        from .lightweight_llm_rag_founders_ai import lightweight_llm_rag
        res = await lightweight_llm_rag.process_query(query=query, top_k=top_k)
        return {"response": res.get("response", ""), "sources": res.get("sources", [])}
    except Exception as e:
        logger.debug(f"search_startup_case_studies failed: {e}")
        return {"response": "", "sources": []}


# -----------------------
# Fallback agent (used if LangChain agent APIs unavailable)
# -----------------------
class FallbackAgent:
    """Simple agent: run RAG -> extract metrics/sector -> synthesize via LLM."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = LLMClient(model=model)
        self.model = model

    async def run(self, query: str) -> FounderResearchOutput:
        rag = await search_startup_case_studies(query, top_k=5)
        rag_text = rag.get("response", "") or ""
        rag_sources = rag.get("sources", []) or []

        metrics = extract_startup_metrics(rag_text or query)
        sector = classify_startup_sector(query)

        system = "You are an expert startup research assistant. Produce a concise summary and 2-6 actionable bullet points."
        user = (
            f"Query: {query}\n\n"
            f"RAG excerpt: {(rag_text[:400] + '...') if rag_text else 'No RAG content.'}\n\n"
            f"Metrics: {', '.join(metrics) if metrics else 'None'}\n"
            f"Sector: {sector.get('category')} (confidence {sector.get('confidence')})\n\n"
            "Output: 1-line summary, then 2-6 bullet points. Keep concise."
        )

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        try:
            answer = await self.llm.generate(messages=messages, max_tokens=600)
        except Exception as e:
            logger.error(f"LLM generation error in fallback agent: {e}")
            answer = "Error generating answer."

        return FounderResearchOutput(
            response=answer.strip(),
            citations=metrics,
            sector=sector.get("category", "Other"),
            confidence=float(sector.get("confidence", 0.0)),
            sources=rag_sources,
            tools_used=["fallback_agent", "lightweight_rag"]
        )


# -----------------------
# LangChain-agent adapter (if APIs available)
# -----------------------
class LangChainAgentAdapter:
    """
    Adapter that builds a LangChain agent with create_openai_tools_agent + AgentExecutor.
    If the requested LangChain functions are not present at runtime, this adapter will not be used.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.llm_client = LLMClient(model=model)
        self.agent_executor = None
        self._built = False

    async def build(self):
        """Build the LangChain agent executor if create_openai_tools_agent is available."""
        if not LANGCHAIN_AGENT_AVAILABLE:
            raise RuntimeError("LangChain agent APIs not available.")

        if self._built:
            return

        # Build tool wrappers compatible with expected LangChain tool API.
        # Instead of importing heavy LangChain tooling here, we construct simple callables
        # that the user's create_openai_tools_agent may accept.
        def metrics_tool_fn(text: str) -> str:
            return ", ".join(extract_startup_metrics(text) or ["No metrics"])

        def sector_tool_fn(text: str) -> str:
            s = classify_startup_sector(text)
            return f"{s.get('category')}|{s.get('confidence')}"

        async def rag_tool_fn(query: str) -> str:
            out = await search_startup_case_studies(query, top_k=5)
            return out.get("response", "")

        # If create_openai_tools_agent is missing → do not attempt to build agent
        if create_openai_tools_agent is None:
            logger.warning("create_openai_tools_agent is not available in this LangChain version.")
            self.agent_executor = None
            self._built = False
            return

        try:
            agent = create_openai_tools_agent(
                llm=self.llm_client._langchain_client,
                tools=[
                    ("extract_startup_metrics", metrics_tool_fn, "Extract metrics"),
                    ("classify_startup_sector", sector_tool_fn, "Classify sector"),
                    ("search_startup_case_studies", rag_tool_fn, "Search case studies"),
                ],
                prompt=None,
            )

            if AgentExecutor is None:
                logger.warning("AgentExecutor not available. Falling back.")
                self.agent_executor = None
                self._built = False
                return

            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=[],
                verbose=False
            )
            self._built = True

        except Exception as e:
            logger.warning(f"Failed to build LangChain agent: {e}")
            self.agent_executor = None
            self._built = False

    async def run(self, query: str) -> FounderResearchOutput:
        if not self._built:
            await self.build()

        if self.agent_executor:
            try:
                # Some AgentExecutor implementations expose .run / .invoke; try both.
                if hasattr(self.agent_executor, "run"):
                    res = await maybe_await(self.agent_executor.run(query))
                elif hasattr(self.agent_executor, "invoke"):
                    res = await maybe_await(self.agent_executor.invoke({"input": query}))
                else:
                    res = None
                # Try to extract textual output
                if isinstance(res, dict):
                    answer = res.get("output") or res.get("result") or str(res)
                else:
                    answer = str(res)
                # Still call RAG and metrics to populate structured return
                rag = await search_startup_case_studies(query, top_k=5)
                metrics = extract_startup_metrics(answer)
                sector = classify_startup_sector(query)
                return FounderResearchOutput(
                    response=answer,
                    citations=metrics,
                    sector=sector.get("category", "Other"),
                    confidence=float(sector.get("confidence", 0.0)),
                    sources=rag.get("sources", []),
                    tools_used=["langchain_agent"]
                )
            except Exception as e:
                logger.error(f"LangChain agent run error: {e}")

        # If anything fails, fallback to the simple agent
        fallback = FallbackAgent(model=self.model)
        return await fallback.run(query)


# -----------------------
# Helpers
# -----------------------
async def maybe_await(value):
    if asyncio.iscoroutine(value):
        return await value
    return value


# -----------------------
# Single exported instance (backwards compatible)
# -----------------------
# Use LangChainAgentAdapter only if langchain agent APIs truly exist (best-effort).
if LANGCHAIN_AGENT_AVAILABLE:
    langchain_startup_agent = LangChainAgentAdapter()
    langchain_legal_agent = langchain_startup_agent
else:
    # Provide a simple agent instance with fallback behavior
    simple_agent_instance = FallbackAgent()
    langchain_startup_agent = simple_agent_instance
    langchain_legal_agent = simple_agent_instance

# Export names expected by other parts of your codebase:
__all__ = [
    "langchain_startup_agent",
    "langchain_legal_agent",
    "FounderQueryInput",
    "FounderResearchOutput",
]
