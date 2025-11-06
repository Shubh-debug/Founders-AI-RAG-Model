"""
Prompt templates for Founders AI founder queries with intent-based response generation.

Provides structured prompts for different query intents (definition, list, explanation, etc)
with startup context integration and response guidelines for founder-focused answers.
"""

from __future__ import annotations
import re
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TypedDict, List
from functools import lru_cache
import yaml
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Unified QueryIntent Enum ---
from enum import Enum
class QueryIntent(str, Enum):
        DEFINITION = "definition"
        LIST = "list"
        EXPLANATION = "explanation"
        COMPARATIVE = "comparative"
        PROCEDURAL = "procedural"
        ANALYTICAL = "analytical"
        INTERPRETATIVE = "interpretative"
        FACTUAL = "factual"


# --- Types ---
class PromptDict(TypedDict):
    system: str
    user: str


@dataclass
class PromptTemplate:
    intent: QueryIntent
    system_prompt: str
    user_template: str
    response_guidelines: str
    max_tokens: int = 256
    temperature: float = 0.2
    model: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def generate_user_content(
        self,
        query: str,
        context: str,
        conflict_info: str = "",
        tone: Optional[str] = None,
        depth: Optional[str] = None
    ) -> str:
        """Generate the user content text for the LLM."""
        tone_instruction = ""
        if tone:
            tone_instruction += f"\nTone: {tone.strip().capitalize()}."
        if depth:
            tone_instruction += f" Depth: {depth.strip().lower()}."

        guidelines = self.response_guidelines + tone_instruction

        mapping = {
            "query": query,
            "context": context,
            "conflict_info": conflict_info,
            "guidelines": guidelines,
        }
        try:
            return self.user_template.format_map(mapping)
        except Exception as e:
            logger.exception("Template formatting failed. Falling back to safe concatenation: %s", e)
            return (
                f"{guidelines}\n\nContext:\n{context}\n\nConflict Info:\n{conflict_info}\n\nQuestion: {query}"
            )


class PromptTemplateManager:
    """Manages startup-specific prompt templates based on query intent."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.templates = self._load_from_yaml(config_path)
        else:
            self.templates = self._default_templates()
        self.validate_templates()
        self._generate_cache = lru_cache(maxsize=256)(self._generate_prompt_cached)

    # ----------------------
    # Loading / Defaults
    # ----------------------
    def _load_from_yaml(self, path: str) -> Dict[QueryIntent, PromptTemplate]:
        """Load templates from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        templates: Dict[QueryIntent, PromptTemplate] = {}
        for key, value in raw.items():
            try:
                intent = self._intent_from_key(key)
                pt = PromptTemplate(
                    intent=intent,
                    system_prompt=value["system_prompt"],
                    user_template=value["user_template"],
                    response_guidelines=value.get("response_guidelines", ""),
                    max_tokens=int(value.get("max_tokens", 256)),
                    temperature=float(value.get("temperature", 0.2)),
                    model=value.get("model"),
                    tags=value.get("tags", []),
                )
                templates[intent] = pt
            except Exception as e:
                logger.exception("Failed to parse template for key %s: %s", key, e)
                raise
        logger.info("Loaded %d prompt templates from %s", len(templates), path)
        return templates

    def _default_templates(self) -> Dict[QueryIntent, PromptTemplate]:
        """Default startup-specific templates aligned with QueryIntent."""
        defaults: Dict[QueryIntent, PromptTemplate] = {}

        def add_template(intent: QueryIntent, sys: str, user: str, guide: str, max_t: int, temp: float, model: str):
            defaults[intent] = PromptTemplate(
                intent=intent,
                system_prompt=sys,
                user_template=user,
                response_guidelines=guide,
                max_tokens=max_t,
                temperature=temp,
                model=model,
            )

        # --- Definition ---
        add_template(
            QueryIntent.DEFINITION,
            "You are a startup research assistant who defines startup concepts concisely using case study data.",
            "{guidelines}\n\nContext from Startup Case Studies:\n{context}{conflict_info}\n\nFounder Question: {query}",
            (
                "RESPONSE REQUIREMENTS:\n"
                "- Define clearly in 2-3 sentences\n"
                "- Include example from startup context if available\n"
                "- Only use factual case study info"
            ),
            200,
            0.1,
            "gpt-4o-mini",
        )

        # --- List ---
        add_template(
            QueryIntent.LIST,
            "You are a startup research assistant creating structured lists based on startup case studies.",
            "{guidelines}\n\nContext from Startup Case Studies:\n{context}{conflict_info}\n\nFounder Question: {query}",
            (
                "RESPONSE REQUIREMENTS:\n"
                "- Provide numbered list (max 10)\n"
                "- Include company examples with metrics if in context\n"
                "- Only use info present in context"
            ),
            300,
            0.2,
            "gpt-4o-mini",
        )

        # --- Explanation ---
        add_template(
            QueryIntent.EXPLANATION,
            "You are a startup strategy analyst explaining business concepts with case study backing.",
            "{guidelines}\n\nContext from Startup Case Studies:\n{context}{conflict_info}\n\nFounder Question: {query}",
            (
                "RESPONSE REQUIREMENTS:\n"
                "- Provide detailed structured explanation with examples\n"
                "- Use case studies with ARR/MAU/funding data\n"
                "- Focus on real founder insights"
            ),
            600,
            0.3,
            "gpt-4o",
        )

        # --- Comparative ---
        add_template(
            QueryIntent.COMPARATIVE,
            "You are a startup analyst comparing strategies or metrics across companies.",
            "{guidelines}\n\nContext from Startup Case Studies:\n{context}{conflict_info}\n\nFounder Question: {query}",
            (
                "RESPONSE REQUIREMENTS:\n"
                "- Compare at least two startups side-by-side\n"
                "- Include metrics (ARR, MAU, funding)\n"
                "- Conclude with founder insights"
            ),
            700,
            0.3,
            "gpt-4o",
        )

        # --- Procedural ---
        add_template(
            QueryIntent.PROCEDURAL,
            "You are a startup mentor explaining step-by-step processes grounded in real examples.",
            "{guidelines}\n\nContext from Startup Case Studies:\n{context}{conflict_info}\n\nFounder Question: {query}",
            (
                "RESPONSE REQUIREMENTS:\n"
                "- Provide steps clearly numbered\n"
                "- Add company examples when relevant\n"
                "- Focus on practical implementation"
            ),
            500,
            0.2,
            "gpt-4o",
        )

        # --- Analytical ---
        add_template(
            QueryIntent.ANALYTICAL,
            "You are a startup research analyst providing deep analytical insights based on data and trends.",
            "{guidelines}\n\nContext from Startup Case Studies:\n{context}{conflict_info}\n\nFounder Question: {query}",
            (
                "RESPONSE REQUIREMENTS:\n"
                "- Analyze patterns and trends\n"
                "- Use data points from startups (growth, users, funding)\n"
                "- Provide strategic implications"
            ),
            800,
            0.4,
            "gpt-4o",
        )

        # --- Interpretative ---
        add_template(
            QueryIntent.INTERPRETATIVE,
            "You are a startup advisor interpreting business patterns and explaining their implications for founders.",
            "{guidelines}\n\nContext from Startup Case Studies:\n{context}{conflict_info}\n\nFounder Question: {query}",
            (
                "RESPONSE REQUIREMENTS:\n"
                "- Interpret data and explain significance\n"
                "- Link insights to founder strategy\n"
                "- Include key takeaways"
            ),
            600,
            0.3,
            "gpt-4o",
        )

        # --- Factual ---
        add_template(
            QueryIntent.FACTUAL,
            "You are a startup data assistant providing direct, factual answers using startup metrics.",
            "{guidelines}\n\nContext from Startup Case Studies:\n{context}{conflict_info}\n\nFounder Question: {query}",
            (
                "RESPONSE REQUIREMENTS:\n"
                "- Provide concise factual responses\n"
                "- Include startup name + metrics\n"
                "- Only use verified data from context"
            ),
            250,
            0.1,
            "gpt-4o-mini",
        )

        logger.info("Initialized %d Founders AI templates (aligned with QueryIntent)", len(defaults))
        return defaults

    # ----------------------
    # Helpers
    # ----------------------
    def _intent_from_key(self, key: str) -> QueryIntent:
        """Match intent regardless of case or value style."""
        key_str = str(key).strip().lower()
        for member in QueryIntent:
            if key_str in {member.name.lower(), member.value.lower()}:
                return member
        raise KeyError(f"Unknown QueryIntent key: {key}")

    def validate_templates(self) -> None:
        """Ensure templates have required fields."""
        missing = [i for i, t in self.templates.items() if not (t.system_prompt and t.user_template)]
        if missing:
            raise ValueError(f"Templates missing required fields for intents: {missing}")
        logger.info("All templates validated successfully.")

    def refine_context(self, context: str) -> str:
        """Normalize and truncate long contexts."""
        if not context:
            return ""
        ctx = re.sub(r"\r\n", "\n", context.strip())
        ctx = re.sub(r"\n{2,}", "\n\n", ctx)
        if len(ctx) > 12000:
            ctx = ctx[:6000] + "\n\n...[context truncated]...\n\n" + ctx[-6000:]
        return ctx

    # ----------------------
    # Prompt generation + caching
    # ----------------------
    def get_template(self, intent: QueryIntent) -> PromptTemplate:
        """Retrieve template for given intent, fallback to factual."""
        return self.templates.get(intent, self.templates[QueryIntent.FACTUAL])

    def get_generation_parameters(self, intent: QueryIntent) -> Dict[str, Any]:
        t = self.get_template(intent)
        return {"model": t.model, "max_tokens": t.max_tokens, "temperature": t.temperature}

    def generate_prompt(
        self,
        intent: QueryIntent,
        query: str,
        context: str,
        conflict_info: str = "",
        tone: Optional[str] = None,
        depth: Optional[str] = None,
    ) -> PromptDict:
        """Generate a complete (system + user) prompt, cached for performance."""
        refined = self.refine_context(context or "")
        context_hash = hashlib.sha256(refined.encode("utf-8")).hexdigest()
        return self._generate_cache(
            intent.value, query, context_hash, refined, conflict_info or "", tone or "", depth or ""
        )

    def _generate_prompt_cached(
        self,
        intent_value: str,
        query: str,
        context_hash: str,
        refined_context: str,
        conflict_info: str,
        tone: str,
        depth: str,
    ) -> PromptDict:
        intent = self._intent_from_key(intent_value)
        template = self.get_template(intent)
        user_content = template.generate_user_content(
            query=query,
            context=refined_context,
            conflict_info=("\n\nConflict Info:\n" + conflict_info) if conflict_info else "",
            tone=tone or None,
            depth=depth or None,
        )
        return {"system": template.system_prompt, "user": user_content}


# Global singleton
prompt_template_manager = PromptTemplateManager()
