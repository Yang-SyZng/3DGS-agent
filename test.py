import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import asyncio
import logging

from workflow.graph import graph

from rag.embedding import embedding
from rag.chunks_builder import flatten_tree, splitter_chunks
from rag.parser.mineru_parser import mineru_parser
from rag.parser.naive_process import (
    build_tree,
    classify_section_tree,
    load_markdown,
    parse_nodes,
)
from rag.vector import MilvusHybridClient

logging.basicConfig(
    filename="y.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


async def main():
    # pdf_path = "test/pdf_samples/FSGS_OCR.pdf"
    paper_id = "FSGS"

    # # 1. PDF -> Markdown
    # parse_result = mineru_parser.parse_pdf(
    #     pdf_path,
    #     backend="hybrid-engine",
    #     extra_args=["-m", "txt", "-high"],
    # )
    # markdown_path = parse_result.get("markdown_path")
    # if not markdown_path:
    #     raise RuntimeError("MinerU 未生成 Markdown 文件")

    # # 2. Markdown -> section tree -> chunks
    # md_doc = load_markdown(markdown_path, metadata={"paper_id": paper_id.lower()})
    # front_node, section_nodes = parse_nodes(md_doc, paper_id=paper_id)
    # roots = classify_section_tree(build_tree(section_nodes, paper_id))
    # flat_nodes = flatten_tree(roots)
    # if front_node is not None:
    #     flat_nodes.append(front_node)

    # text_nodes = splitter_chunks(flat_nodes, paper_id=paper_id)
    # print(f"生成 {len(text_nodes)} 个文本块")

    # # 3. Dense embedding + Milvus BM25 sparse indexing
    # embedded_nodes = embedding.embed_nodes(text_nodes)
    milvus = MilvusHybridClient()
    # ids = milvus.add_documents(embedded_nodes)
    # print(f"写入 Milvus {len(ids)} 个文本块")

    # 4. Native dense + BM25 hybrid retrieval + RRF reranking
    # results = await milvus.search(
    #     "What is the core method of FSGS?",
    #     embed_model=embedding,
    #     filters={"paper_id": [paper_id.lower()]},
    #     top_k=5,
    # )
    # for rank, item in enumerate(results, start=1):
    #     print(
    #         f"[{rank}] score={item.score:.4f} "
    #         f"section={item.node.metadata.get('section_title')}\n"
    #         f"{item.node.text[:300]}\n"
    #     )

    result = await graph.ainvoke({
        "user_query": "FSGS 的实验结果"
    })

    print(result["research_result"])
if __name__ == "__main__":
    asyncio.run(main())
