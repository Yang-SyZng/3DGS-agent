from __future__ import annotations

__all__ = [
    "ArxivQuery",
    "query",
    "ArxivQueryTools",
    "PDFProcessTools",
    "pdf_process",
    "zoteroclient",
    "ZoteroClientTools",
]


def __getattr__(name: str):
    if name in {"ArxivQuery", "query", "ArxivQueryTools"}:
        from .arxiv_query import ArxivQuery, ArxivQueryTools, query

        return {
            "ArxivQuery": ArxivQuery,
            "query": query,
            "ArxivQueryTools": ArxivQueryTools,
        }[name]

    if name in {"PDFProcessTools", "pdf_process"}:
        from .pdf_process import PDFProcessTools, pdf_process

        return {
            "PDFProcessTools": PDFProcessTools,
            "pdf_process": pdf_process,
        }[name]

    if name in {"zoteroclient", "ZoteroClientTools"}:
        from .zotero_query import ZoteroClientTools, zoteroclient

        return {
            "zoteroclient": zoteroclient,
            "ZoteroClientTools": ZoteroClientTools,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
