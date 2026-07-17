from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Any, Mapping

from llama_index.core import PromptTemplate
from llama_index.llms.openai_like import OpenAILike

try:
    from llama_index.llms.ollama import Ollama
except ImportError as exc:
    raise ImportError(
        "Not Module name 'ollama'"
    ) from exc

from config.settings import setting
from prompts.prompts import MatcherPrompt
from schema.matcher_schema import MatchStatus, PaperMatchResult
from tools.mcp_service.paper_service import PaperMCPClient


class PaperMatcher:
    def __init__(
        self,
        llm: OpenAILike | Ollama = None,
        paper_client: PaperMCPClient | None = None,
    ):
        llm_model = OpenAILike(
            api_base=setting.BASE_URL,
            api_key=setting.API_KEY,
            model=setting.LLM_MODEL_ID,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=128000,
        )

        self.llm = llm or llm_model
        self.paper_client = paper_client or PaperMCPClient(timeout=300)
        self.prompt = PromptTemplate(MatcherPrompt)

    async def match(
        self,
        target_paper: str,
        max_results: int = 10,
        fallback_sources: Sequence[str] | None = None,
    ) -> PaperMatchResult:
        if not target_paper.strip():
            raise ValueError("target_paper 不能为空")
        if max_results < 1:
            raise ValueError("max_results 必须大于 0")

        sources = ["arxiv"]
        sources.extend(
            source.strip().lower()
            for source in (fallback_sources or [])
            if source.strip() and source.strip().lower() != "arxiv"
        )

        errors = []
        for source in dict.fromkeys(sources):
            try:
                candidates = await self._search_source(
                    source=source,
                    query=target_paper,
                    max_results=max_results,
                )
            except Exception as exc:
                errors.append(f"{source}: {type(exc).__name__}: {exc}")
                continue

            result = await self.match_paper_metadata(
                target_paper=target_paper,
                candidates=candidates['result'],
                source=source,
            )
            if result.status == MatchStatus.MATCHED:
                return result

        reason = "No searched source returned a reliable metadata match."
        if errors:
            reason = f"{reason} Search errors: {'; '.join(errors)}"
        return PaperMatchResult(
            target_paper=target_paper,
            status=MatchStatus.UNMATCHED,
            confidence=0.0,
            reason=reason,
        )

    async def _search_source(
        self,
        source: str,
        query: str,
        max_results: int,
    ) -> Any:
        if source == "arxiv":
            return await self.paper_client.search_arxiv(
                query=query,
                max_results=max_results,
            )

        return await self.paper_client.search_source(
            source=source,
            query=query,
            max_results=max_results,
        )

    async def download(
        self,
        match_result: PaperMatchResult,
        save_path: str | Path | None = None,
        use_scihub: bool = False,
    ) -> Any:
        if match_result.status != MatchStatus.MATCHED:
            raise ValueError("只能下载状态为 matched 的论文")
        if not match_result.source:
            raise ValueError("匹配结果缺少 source")

        paper_id = match_result.paper_id or match_result.doi
        if not paper_id:
            raise ValueError("匹配结果缺少 paper_id 或 DOI")

        return await self.paper_client.download(
            source=match_result.source,
            paper_id=paper_id,
            doi=match_result.doi or "",
            title=match_result.title or match_result.target_paper,
            save_path=save_path,
            use_scihub=use_scihub,
        )

    async def match_paper_metadata(
        self,
        target_paper: str,
        candidates: Dict | None,
        source: str = "arxiv",
    ) -> PaperMatchResult:
        if not target_paper.strip():
            raise ValueError("target_paper 不能为空")

        if not candidates:
            return PaperMatchResult(
                target_paper=target_paper,
                status=MatchStatus.UNMATCHED,
                confidence=0.0,
                reason="The paper search returned no candidate metadata.",
            )

        result = await self.llm.astructured_predict(
            output_cls=PaperMatchResult,
            prompt=self.prompt,
            target_paper=target_paper,
            source=source,
            candidates=json.dumps(
                candidates,
                ensure_ascii=False,
                default=str,
            ),
        )

        return self._validate_result(
            result,
            target_paper,
            candidates,
            source,
        )

    @staticmethod
    def _validate_result(
        result: PaperMatchResult,
        target_paper: str,
        candidates: list[dict[str, Any]],
        default_source: str,
    ) -> PaperMatchResult:
        result.target_paper = target_paper

        if result.status != MatchStatus.MATCHED or result.confidence < 0.8:
            result.status = MatchStatus.UNMATCHED
            result.candidate_index = None
            result.source = None
            result.paper_id = None
            result.title = None
            result.doi = None
            result.confidence = min(result.confidence, 0.49)
            return result

        index = result.candidate_index
        if index is None or not 0 <= index < len(candidates):
            return PaperMatchResult(
                target_paper=target_paper,
                status=MatchStatus.UNMATCHED,
                confidence=0.0,
                reason="The model did not select a valid candidate index.",
            )

        candidate = candidates[index]
        result.source = str(candidate.get("source") or default_source)
        result.paper_id = PaperMatcher._first_value(
            candidate,
            "paper_id",
            "arxiv_id",
            "id",
        )
        result.title = PaperMatcher._first_value(candidate, "title", "name")
        result.authors = PaperMatcher._first_value(candidate, "authors")
        result.abstract = PaperMatcher._first_value(candidate, "abstract")
        result.published_date = PaperMatcher._first_value(candidate, "published_date")
        result.pdf_url = PaperMatcher._first_value(candidate, "pdf_url")
        result.categories = PaperMatcher._first_value(candidate, "categories")

        return result

    @staticmethod
    def _first_value(candidate: Mapping[str, Any], *keys: str) -> str | None:
        for key in keys:
            value = candidate.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return None
