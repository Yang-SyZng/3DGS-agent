from __future__ import annotations

__all__ = [
    "ArxivQuery",
    "query",
    "ArxivQueryTools",
    "PDFProcessTools",
    "pdf_process",
    "MinerUPDFProcess",
    "MinerUPDFProcessTools",
    "mineru_pdf_process",
    "zoteroclient",
    "ZoteroClientTools",
]


def __getattr__(name: str):
    if name in {"ArxivQuery", "query", "ArxivQueryTools"}:
        from .arxiv_query import ArxivQuery, ArxivQueryTools, query

        values = {
            "ArxivQuery": ArxivQuery,
            "query": query,
            "ArxivQueryTools": ArxivQueryTools,
        }
        globals().update(values)
        return values[name]

    if name in {"PDFProcessTools", "pdf_process"}:
        from .pdf_process import PDFProcessTools, pdf_process

        values = {
            "PDFProcessTools": PDFProcessTools,
            "pdf_process": pdf_process,
        }
        globals().update(values)
        return values[name]

    if name in {"MinerUPDFProcess", "MinerUPDFProcessTools", "mineru_pdf_process"}:
        from .mineru_pdf_process import (
            MinerUPDFProcess,
            MinerUPDFProcessTools,
            mineru_pdf_process,
        )

        values = {
            "MinerUPDFProcess": MinerUPDFProcess,
            "MinerUPDFProcessTools": MinerUPDFProcessTools,
            "mineru_pdf_process": mineru_pdf_process,
        }
        globals().update(values)
        return values[name]

    if name in {"zoteroclient", "ZoteroClientTools"}:
        from .zotero_query import ZoteroClientTools, zoteroclient

        values = {
            "zoteroclient": zoteroclient,
            "ZoteroClientTools": ZoteroClientTools,
        }
        globals().update(values)
        return values[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
