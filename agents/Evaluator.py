from __future__ import annotations

from prompts.prompts import EvaluatorPrompt
from schema.evaluator_schema import RetrievalEvaluation
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import PromptTemplate

import json
from config.settings import setting


class RetrievalEvaluator:
    def __init__(self, llm: OpenAILike = None):
        llm_model = OpenAILike(
            api_base=setting.BASE_URL,
            api_key=setting.API_KEY,
            model=setting.LLM_MODEL_ID,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=128000,
        )

        self.llm = llm_model or llm

        self.prompt = PromptTemplate(EvaluatorPrompt)

    async def evaluate(self,
                query: str,
                analysis,
                retrieved_nodes,
            ) -> RetrievalEvaluation:
        if not retrieved_nodes:
            return RetrievalEvaluation(
                sufficient=False,
                missing_papers=analysis.paper_names,
                missing_information=[
                    "No relevant evidence was retrieved"
                ],
                relevant_chunk_ids=[],
                reason="The retriever returned no evidence.",
            )
        contexts = self._format_contexts(retrieved_nodes)
        result = await self.llm.astructured_predict(
            output_cls=RetrievalEvaluation,
            prompt=self.prompt,
            query=query,
            analysis=json.dumps(
                analysis.model_dump(mode="json"),
                ensure_ascii=False,
            ),
            contexts=contexts,
        )

        result.original_query = query

        return result
    
    def _format_contexts(self, nodes) -> str:
        blocks = []

        for index, item in enumerate(nodes, start=1):
            
            node = item.node if hasattr(item, "node") else item
            score = getattr(item, "score", None)

            chunk_id = node.metadata.get(
                "chunk_id",
                node.metadata.get('chunk_id'),
            )

            block = (
                f"[Chunk {index}]\n"
                f"chunk_id: {chunk_id}\n"
                f"score: {score}\n"
                f"paper_id: {node.metadata.get('paper_id')}\n"
                f"section_path: {node.metadata.get('section_path')}\n"
                f"semantic_type: {node.metadata.get('semantic_type')}\n"
                f"text: {node.text}\n"
            )

            blocks.append(block)

        return "\n".join(blocks)
