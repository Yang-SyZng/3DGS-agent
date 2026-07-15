import os
import subprocess
import time
from urllib.request import urlopen
from typing import Dict

from ollama import Client

try:
    from llama_index.llms.ollama import Ollama
except ImportError as exc:
    raise ImportError(
        "Not Module name 'ollama'"
    ) from exc

class OllamaServer:
    def __init__(self, local_models: Dict):
        self.local_models = local_models
        self.processes: Dict[str, subprocess.Popen] = {}
        self.clients: Dict[str, Client] = {}
        self.llms: Dict[str, Ollama] = {}
        self.start_all()

    @staticmethod
    def _is_server_ready(base_url: str) -> bool:
        try:
            with urlopen(f"{base_url}/api/tags", timeout=2):
                return True
        except Exception:
            return False

    def _connect_instance(self, name: str, config: Dict, base_url: str):
        self.clients[name] = Client(host=base_url)
        self.llms[name] = self.create_ollama_llm(name, base_url=base_url)
        print(
            f"{name} 已连接：GPU {config['gpu']}，"
            f"端口 {config['port']}"
        )
        return self.clients[name]

    def create_ollama_llm(self, instance: str, base_url: str | None = None) -> Ollama:
        config = self.local_models[instance]
        base_url = base_url or f"http://127.0.0.1:{config['port']}"
        return Ollama(
            model=config["model"],
            base_url=base_url,
            context_window=int(config["context_length"]),
            temperature=0.01,
            keep_alive="30m",
            thinking=False,
            json_mode=True,
            is_function_calling_model=False,
            request_timeout=120.0
        )

    def start_instance(self,
                       name: str,
                       config: Dict,
                       timeout: int = 60
        ):
        host = f"127.0.0.1:{config['port']}"
        base_url = f"http://{host}"

        if self._is_server_ready(base_url):
            print(f"复用已有 Ollama 服务：{base_url}")
            return self._connect_instance(name, config, base_url)

        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": config["gpu"],

                "OLLAMA_HOST": host,

                "OLLAMA_MAX_LOADED_MODELS": "1",

                "OLLAMA_NUM_PARALLEL": str(config["num_parallel"]),

                "OLLAMA_CONTEXT_LENGTH": str(config["context_length"]),

                "OLLAMA_MODELS": config["model_dir"],
            }
        )

        process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        self.processes[name] = process

        deadline = time.time() + timeout

        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Ollama 实例 {name} 启动失败，退出码："
                    f"{process.returncode}"
                )

            try:
                with urlopen(f"{base_url}/api/tags", timeout=2):
                    print(f"已启动共享 Ollama 服务：{base_url}")
                    return self._connect_instance(name, config, base_url)
            except Exception:
                time.sleep(1)

        process.terminate()
        raise TimeoutError(f"Ollama 实例 {name} 启动超时")

    def ensure_models(self):
        for name, config in self.local_models.items():
            model = config["model"]
            local_models = {
                item.model for item in self.clients[name].list().models
            }

            if model in local_models:
                print(f"模型已存在：{model}")
                continue

            print(f"正在下载模型：{model}")
            self.clients[name].pull(model=model)

    def warmup_models(self):
        for name, config in self.local_models.items():
            model = config["model"]
            print(f"正在预热模型：{model}")
            started_at = time.perf_counter()
            self.clients[name].generate(
                model=model,
                prompt="",
                think=False,
                keep_alive="30m",
                options={"num_ctx": int(config["context_length"])},
            )
            elapsed = time.perf_counter() - started_at
            print(f"模型预热完成，耗时 {elapsed:.1f} 秒")

    def start_all(self):
        for name, config in self.local_models.items():
            self.start_instance(
                    name=name,
                    config=config
                )
        self.ensure_models()
        self.warmup_models()

    def stop_all(self):
        for name, process in self.processes.items():
            if process.poll() is None:
                print(f"正在关闭 Ollama 实例：{name}")
                process.terminate()

                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        self.processes.clear()