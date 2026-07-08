from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.core.schema import BaseNode, TextNode, RelatedNodeInfo, NodeRelationship

from config import setting
import logging


logger = logging.getLogger(__name__)

class PDFSplitter:
    def __init__(self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        paragraph_separator: str = "\n\n",
    ):
        self.chunk_size = setting.chunk_size if chunk_size is None else chunk_size
        self.chunk_overlap = setting.chunk_overlap if chunk_overlap is None else chunk_overlap
        self.paragraph_separator = paragraph_separator

        self.markdown_parser = MarkdownNodeParser()

        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator=self.paragraph_separator,
        )

        logger.info(
            f"初始化文本切分器: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def load_markdown(self, path: str, metadata: dict | None = None) -> Document:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        return Document(
            text=text,
            metadata=metadata or {},
        )
    
    def SplitByParagraph(
        self,
        nodes: list[BaseNode],
        paragraph_separator: str | None = None,
    ) -> list[TextNode]:
        """按段落分隔符切分节点，并保留原节点 metadata 和 SOURCE 关系。"""
        separator = paragraph_separator or self.paragraph_separator
        paragraph_nodes: list[TextNode] = []
        paragraph_index = 0

        for node in nodes:
            text = node.get_content()
            parts = [part.strip() for part in text.split(separator) if part.strip()]

            for local_paragraph_index, part in enumerate(parts):
                relationships = dict(node.relationships)
                relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=node.node_id)
                paragraph_nodes.append(
                    TextNode(
                        text=part,
                        metadata={
                            **node.metadata,
                            "paragraph_index": paragraph_index,
                            "local_paragraph_index": local_paragraph_index,
                        },
                        relationships=relationships,
                    ),
                )
                paragraph_index += 1

        logger.info("段落切分完成，生成%s个段落节点", len(paragraph_nodes))
        return paragraph_nodes

    def SplitMarkdownDocument(self, document: Document) -> list[BaseNode]:
        markdown_nodes = self.markdown_parser.get_nodes_from_documents([document])
        paragraph_nodes = self.SplitByParagraph(markdown_nodes)
        nodes = self.splitter.get_nodes_from_documents(paragraph_nodes)
        logger.info("Markdown文档切分完成，生成%s个节点", len(nodes))
        return nodes

    def SplitMarkdownFile(
        self,
        path: str,
        metadata: dict | None = None,
    ) -> list[BaseNode]:
        document = self.load_markdown(path, metadata=metadata)
        return self.SplitMarkdownDocument(document)

splitter = PDFSplitter()
