from __future__ import annotations

import json

from llama_index.core import PromptTemplate
from llama_index.llms.openai_like import OpenAILike

try:
    from llama_index.llms.ollama import Ollama
except ImportError as exc:
    raise ImportError("Not Module name 'ollama'") from exc

from config.settings import setting
from prompts.prompts import WriterPrompt
from schema.analyzer_schema import QueryAnalysis
from schema.evaluator_schema import RetrievalEvaluation


class AnswerWriter:
    def __init__(self, llm: OpenAILike | Ollama = None):
        default_llm = OpenAILike(
            api_base=setting.BASE_URL,
            api_key=setting.API_KEY,
            model=setting.LLM_MODEL_ID,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=128000,
        )
        self.llm = llm or default_llm
        self.prompt = PromptTemplate(WriterPrompt)

    async def write(
        self,
        query: str,
        analysis: QueryAnalysis,
        evaluation: RetrievalEvaluation,
        retrieved_nodes: list,
        external_search_results: list | None = None,
        external_search_errors: list[str] | None = None,
    ) -> str:
        contexts = self._format_contexts(
            retrieved_nodes,
            evaluation.relevant_chunk_ids,
        )
        external_context = self._format_external_results(
            external_search_results or [],
            external_search_errors or [],
        )
        if external_context:
            contexts = "\n\n".join(
                part for part in (contexts, external_context) if part
            )
        return await self.llm.apredict(
            self.prompt,
            query=query,
            analysis=json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False),
            evaluation=json.dumps(evaluation.model_dump(mode="json"), ensure_ascii=False),
            contexts=contexts,
        )

    @staticmethod
    def _format_external_results(results: list, errors: list[str]) -> str:
        if not results and not errors:
            return ""
        payload = {
            "results": [
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else item
                for item in results
            ],
            "errors": errors,
        }
        return "External literature search:\n" + json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def _format_contexts(nodes: list, relevant_chunk_ids: list[str]) -> str:
        relevant_ids = {str(chunk_id) for chunk_id in relevant_chunk_ids}
        blocks = []
        for item in nodes:
            node = item.node if hasattr(item, "node") else item
            metadata = getattr(node, "metadata", {}) or {}
            chunk_id = metadata.get("chunk_id")
            if str(chunk_id) not in relevant_ids:
                continue
            blocks.append(
                "\n".join(
                    [
                        f"chunk_id: {chunk_id}",
                        f"paper_id: {metadata.get('paper_id')}",
                        f"paper_title: {metadata.get('paper_title')}",
                        f"section_path: {metadata.get('section_path')}",
                        f"text: {getattr(node, 'text', '')}",
                    ]
                )
            )
        return "\n\n".join(blocks)
