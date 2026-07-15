from __future__ import annotations

from prompts.prompts import ResearchSynthesizerPrompt
from schema.analyzer_schema import QueryAnalysis
from schema.researcher_schema import ResearchResult
from schema.evaluator_schema import RetrievalEvaluation

from llama_index.llms.openai_like import OpenAILike
try:
    from llama_index.llms.ollama import Ollama
except ImportError as exc:
    raise ImportError(
        "Not Module name 'ollama'"
    ) from exc

from llama_index.core import PromptTemplate

import json
from config.settings import setting

class ResearchSynthesizer:
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

        self.prompt = PromptTemplate(ResearchSynthesizerPrompt)
    
    async def synthesis(self,
                    query: str,
                    analysis: QueryAnalysis,
                    retrieved_nodes: list,
                    retrieval_evaluation: RetrievalEvaluation
              ) -> ResearchResult:
        if not retrieved_nodes:
            return ResearchResult(
                limitations=[
                    "No retrieved evidence is available for synthesis."
                ]
            )

        contexts = self._format_contexts(
            retrieved_nodes,
            retrieval_evaluation.relevant_chunk_ids,
        )
        if not contexts:
            return ResearchResult(
                limitations=[
                    retrieval_evaluation.reason,
                    *retrieval_evaluation.missing_information,
                ]
            )

        result = await self.llm.astructured_predict(
            output_cls=ResearchResult,
            prompt=self.prompt,
            original_query=query,
            query_analysis=json.dumps(
                analysis.model_dump(mode="json"),
                ensure_ascii=False,
            ),
            retrieval_evaluation=json.dumps(
                retrieval_evaluation.model_dump(mode="json"),
                ensure_ascii=False,
            ),
            retrieved_evidence=contexts,
        )

        if not retrieval_evaluation.sufficient:
            existing = set(result.limitations)
            for limitation in [
                retrieval_evaluation.reason,
                *retrieval_evaluation.missing_information,
            ]:
                if limitation and limitation not in existing:
                    result.limitations.append(limitation)
                    existing.add(limitation)

        return result

    def _format_contexts(
        self,
        nodes: list,
        relevant_chunk_ids: list[str] | None = None,
    ) -> str:
        relevant_ids = {
            str(chunk_id)
            for chunk_id in (relevant_chunk_ids or [])
            if chunk_id is not None
        }
        blocks = []

        for index, item in enumerate(nodes, start=1):
            node = item.node if hasattr(item, "node") else item
            metadata = getattr(node, "metadata", {}) or {}
            chunk_id = metadata.get("chunk_id")

            if relevant_ids and str(chunk_id) not in relevant_ids:
                continue

            score = getattr(item, "score", None)
            text = getattr(node, "text", "")
            block = (
                f"[Evidence {len(blocks) + 1}]\n"
                f"source_index: {index}\n"
                f"chunk_id: {chunk_id}\n"
                f"score: {score}\n"
                f"paper_id: {metadata.get('paper_id')}\n"
                f"paper_title: {metadata.get('paper_title')}\n"
                f"section_path: {metadata.get('section_path')}\n"
                f"section_title: {metadata.get('section_title')}\n"
                f"section_type: {metadata.get('section_type')}\n"
                f"text: {text}\n"
            )
            blocks.append(block)

        return "\n".join(blocks)
