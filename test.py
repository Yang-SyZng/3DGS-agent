import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import re

from llama_index.core import Document

from rag.splitter import splitter
from rag.embedding import embedding
from rag.vector import milvusvector
logging.basicConfig(
    filename="y.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def clean_pdf_text(text: str) -> str:
    # 1. 统一特殊空白字符
    text = text.replace("\xa0", " ")
    text = text.replace("\u3000", " ")

    # 2. 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. 删除页码类内容，例如：— 1 —、- 1 -
    text = re.sub(r"(?m)^\s*[—\-–]\s*\d+\s*[—\-–]\s*$", "", text)

    # 4. 合并标题中的字间空格
    # 例如：政 府 工 作 报 告 -> 政府工作报告
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)

    # 5. 保留段落分隔：多个换行压缩为两个换行
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # 6. 删除中文句子中的普通换行
    # 例如：请予审议，并请\n市政协 -> 请予审议，并请市政协
    text = re.sub(
        r"(?<=[\u4e00-\u9fff，、；：])\n(?=[\u4e00-\u9fff])",
        "",
        text
    )

    # 7. 英文/数字之间的换行转为空格
    # 例如：machine\nlearning -> machine learning
    text = re.sub(
        r"(?<=[A-Za-z0-9])\n(?=[A-Za-z0-9])",
        " ",
        text
    )

    # 8. 行首行尾空格清理
    lines = [line.strip() for line in text.split("\n")]

    # 9. 去掉空白太多的行
    text = "\n".join(lines)

    # 10. 多个空格合并
    text = re.sub(r"[ \t]+", " ", text)

    # 11. 最后再次压缩多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def main():
    # mineru_pdf_process.parse_pdf(
    #     "test/pdf_samples/text_sample_CN.pdf",
    #     backend="hybrid-engine",
    #     extra_args=["-m", "txt", "-high"],
    #     # effort="high"
    # )

        # 1. 读取md
    doc = splitter.load_markdown("database/pdf_ocr_results/text_sample_CN/text_sample_CN.md")

    node = splitter.SplitMarkdownDocument(doc)

    node = embedding.embed_nodes(node)
    res = milvusvector.add_documents(node)
    
    print(node)


    # print(embedding.embed_text("你是谁"))
    # milvusvector = get_milvus_client()

    # output = extract_text("test/pdf_samples/text_sample_CN.pdf")
    # output = clean_pdf_text(output)
    # chunks = splitter.SplitText(output)
    
    # vector_chunks = [embedding.embed_text(chunk)[0] for chunk in chunks]
    
    # data = [{"id": i, "vector": vector_chunks[i], "text": chunks[i]} for i in range(len(chunks))]

    # milvusvector.insert(
    #     collection_name="arxiv_papers",
    #     data=data
    # )
    # res = milvusvector.search(
    #     collection_name="arxiv_papers",
    #     data=[embedding.embed_query("营商环境创新县市 知识产权运营服务集聚区 社会投资项目用地清单制 二手房带押过户扩展类型 低效用地6500亩盘活 批而未供土地清理 产业社区标准化园区入园率 国企资产2200亿元 AA+评级 国企商业保理牌照 枫桥式治理 永和镇调解 中德科技论坛 达沃市结好 金门供水3100万吨")],
    #     limit=10,
    # )[0]
    # print(res)

    # print(output)
    # print(documents)
    # print(chunks)

if __name__ == "__main__":
    main()
