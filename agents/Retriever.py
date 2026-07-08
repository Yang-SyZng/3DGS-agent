from __future__ import annotations

import logging

from rag.vector import ragtools

from .Baser import BaseFunctionAgent

logger = logging.getLogger(__name__)


class Retriever(BaseFunctionAgent):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "Retriever")
        kwargs.setdefault("description", "Retrieve relevant indexed documents from the vector store.")
        kwargs.setdefault("tools", ragtools)
        super().__init__(*args, **kwargs)
