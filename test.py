import sys
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from agents import DocumentRouterAgent, ZoteroAgent
import logging

logging.basicConfig(
    filename="y.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def main():
    print("Loading...")
    # output = pdf_process.extract_pdf_stats(file_path="database/Papers/2308.04079.pdf")
    # print(output)
    # output = pdf_process.extract_text_with_images(pdf_path="database/Papers/2308.04079.pdf", output_dir="database/output", save_md=True, md_path="database/output/test.md")
    # print(output)
    # agent = DocumentRouterAgent(tools=PDFProcessTools)
    agent = ZoteroAgent()
    response = agent.invoke({"messages": [{"role": "user", "content": "帮我找一下GS-RoadPatching论文，然后下载到database/Cache"}]})

    print(response["messages"][-1].content)

    # stream = agent.stream(
    #     {"messages": [{"role": "user", "content": "帮我判断database/Papers/2308.04079.pdf"}]},
    #     stream_mode="messages",
    #     print_mode=(),
    # )

    # print("Streaming response:")
    # for event in stream:
    #     if isinstance(event, tuple) and len(event) == 2:
    #         mode, data = event
    #         if mode == "messages" and isinstance(data, tuple):
    #             token, _metadata = data
    #             print(token, end="", flush=True)
    #     elif isinstance(event, dict):
    #         # 如果返回完整数据结构，可按需显示
    #         print(event)
    # print()
if __name__ == "__main__":
    main()