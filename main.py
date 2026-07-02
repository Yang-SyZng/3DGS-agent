import sys
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from tools import tools, pdf_process
from agents import build_agent
import logging

logging.basicConfig(
    filename="y.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def main():
    print("Loading...")
    pdf_process.infer()
    # agent = build_agent(tools=tools)

    # response = agent.invoke({"messages": [{"role": "user", "content": "帮我找寻Lin Gao的3DGS方向的最新的3篇论文，并且下载下来"}]})

    # print(response["messages"][-1].content)

if __name__ == "__main__":
    main()