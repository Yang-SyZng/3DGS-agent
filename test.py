import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import asyncio
import logging
from workflow.graph import graph

logging.basicConfig(
    filename="y.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


async def main():
    # mineru_parser.parse_pdf(
    #     "test/pdf_samples/FSGS_OCR.pdf",
    #     backend="hybrid-engine",
    #     extra_args=["-m", "txt", "-high"],
    # )
    # md_doc = load_markdown("database/parser/AbsGS/AbsGS.md")
    # front_node, result = parse_nodes(md_doc)
    # roots = build_tree(result, "AbsGS")
    # roots = classify_section_tree(roots)
    # flatten_roots = flatten_tree(roots)
    # if front_node is not None:
    #     flatten_roots.append(front_node)
    # text_nodes = splitter_chunks(flatten_roots)
    # embed_nodes = embedding.embed_nodes(text_nodes)
    # with open(
    #     "database/parser/AbsGS/data.json",
    #     "r",
    #     encoding="utf-8"
    # ) as f:
    #     embed_nodes = json.load(f)
    # ids = milvusvector.add_documents(embed_nodes)
    # results = milvusvector.search("method")
    # print(results)

    result = await graph.ainvoke({
        "user_query": "AbsGS 的核心方法是什么？"
    })

    print(result["research_result"])

if __name__ == "__main__":
    asyncio.run(main())
