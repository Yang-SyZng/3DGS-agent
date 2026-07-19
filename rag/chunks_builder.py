from typing import List
from schema.paper_schema import SectionNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from config import setting

def flatten_tree(nodes: List[SectionNode]):
    flatten_results: List[SectionNode] = []
    for node in nodes:
        # 当前节点加入
        flatten_results.append(node)

        # 递归children
        if node.children:
            flatten_results.extend(flatten_tree(node.children))

    return flatten_results

def splitter_chunks(nodes: List[SectionNode], paragraph_separator: str = "\n\n", paper_id: str = "unknown",) -> List[TextNode]:
    splitter = SentenceSplitter(
        chunk_size=setting.chunk_size,
        chunk_overlap=setting.chunk_overlap,
        paragraph_separator=paragraph_separator
    )
    chunks_nodes = []
    for node in nodes:
        chunks = splitter.split_text(
            node.content
        )
        for idx, chunk in enumerate(chunks):
            chunks_node = TextNode(
                text=chunk,
                metadata={
                    "chunk_id": f"{node.section_id}:{idx}",
                    "paper_id": paper_id.lower(),
                    "section_id": node.section_id,
                    "section_title": node.title,
                    "level": node.level,
                    "section_path": "/".join(node.path),
                    "section_type": node.semantic_type.value if node.semantic_type else None,
                }
            ).dict()
            chunks_nodes.append(chunks_node)

    return chunks_nodes