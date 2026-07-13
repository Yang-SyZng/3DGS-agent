import os
import subprocess
import time

import requests


OLLAMA_URL = "http://127.0.0.1:11434"


def ollama_is_running() -> bool:
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=1,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def start_ollama() -> subprocess.Popen | None:
    if ollama_is_running():
        print("Ollama已经启动")
        return None

    env = os.environ.copy()

    # 只让Ollama看到22GB显卡
    env["CUDA_VISIBLE_DEVICES"] = "1"

    # 模型常驻显存
    env["OLLAMA_KEEP_ALIVE"] = "-1"

    # 22GB运行27B模型时先设为单并发
    env["OLLAMA_NUM_PARALLEL"] = "1"
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"

    # 先从8K上下文开始
    env["OLLAMA_CONTEXT_LENGTH"] = "8192"

    # 降低KV Cache显存占用
    env["OLLAMA_FLASH_ATTENTION"] = "1"
    env["OLLAMA_KV_CACHE_TYPE"] = "q8_0"

    process = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
    )

    for _ in range(30):
        if ollama_is_running():
            print("Ollama启动成功")
            return process

        if process.poll() is not None:
            raise RuntimeError(
                f"Ollama启动失败，退出码：{process.returncode}"
            )

        time.sleep(1)

    process.terminate()
    raise TimeoutError("等待Ollama启动超时")

def preload_model(
    model: str = "qwen3.5:27b-q4_K_M",
) -> None:
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [],
            "stream": False,
            "keep_alive": -1,
        },
        timeout=300,
    )
    response.raise_for_status()
    print(f"{model} 已加载到显存")
    
if __name__ == "__main__":
    ollama_process = start_ollama()
    preload_model()