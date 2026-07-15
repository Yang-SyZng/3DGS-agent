from __future__ import annotations

from prompts.prompts import AnalyzerPrompt
from schema.analyzer_schema import QueryAnalysis

from llama_index.llms.openai_like import OpenAILike
try:
    from llama_index.llms.ollama import Ollama
except ImportError as exc:
    raise ImportError(
        "Not Module name 'ollama'"
    ) from exc

from llama_index.core import PromptTemplate

from config.settings import setting


class QueryAnalyzer:
    def __init__(self, llm: OpenAILike | Ollama = None):
        llm_model = OpenAILike(
            api_base=setting.BASE_URL,
            api_key=setting.API_KEY,
            model=setting.LLM_MODEL_ID,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=128000,
        )

        self.llm = llm or llm_model

        self.prompt = PromptTemplate(AnalyzerPrompt)

    async def analyze(self, query: str) -> QueryAnalysis:
        result = await self.llm.astructured_predict(
            output_cls=QueryAnalysis,
            prompt=self.prompt,
            query=query,
        )

        # if not result.keywords:
        #     result.keywords = self._build_fallback_keywords(result)

        return result

    # @staticmethod
    # def _build_fallback_keywords(analysis: QueryAnalysis) -> list[str]:
    #     """Build stable retrieval keywords when the LLM returns an empty list."""
    #     target_terms = {
    #         "method": "method",
    #         "experiment": "experimental evaluation",
    #         "comparison": "method comparison",
    #         "summary": "paper summary",
    #         "background": "research background",
    #         "other": "scientific literature",
    #     }
    #     candidates = [*analysis.paper_names, *analysis.entities]
    #     candidates.extend(
    #         target_terms.get(
    #             target.value if hasattr(target, "value") else str(target),
    #             str(target),
    #         )
    #         for target in analysis.targets
    #     )

    #     keywords: list[str] = []
    #     seen: set[str] = set()
    #     for candidate in candidates:
    #         keyword = candidate.strip()
    #         normalized = keyword.casefold()
    #         if not keyword or normalized in seen:
    #             continue
    #         keywords.append(keyword)
    #         seen.add(normalized)
    #         if len(keywords) == 8:
    #             break

    #     return keywords or ["scientific literature"]
