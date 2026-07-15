import asyncio

from prompts.prompts import AnalyzerPrompt
from schema.analyzer_schema import QueryAnalysis
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import PromptTemplate

from tools.llm_local_service.ollama_service import OllamaServer

from config.settings import setting

try:
    from llama_index.llms.ollama import Ollama
except ImportError as exc:
    raise ImportError(
        "Not Module name 'ollama'"
    ) from exc

class QueryAnalyzer:
    def __init__(self, llm: Ollama = None):
        llm_model = OpenAILike(
            api_base=setting.BASE_URL,
            api_key=setting.API_KEY,
            model=setting.LLM_MODEL_ID,
            is_chat_model=True,
            is_function_calling_model=False,
            context_window=128000,
        )
        self.llm = llm or llm_model

        self.prompt = PromptTemplate(AnalyzerPrompt)

    async def analyze(self, query: str) -> QueryAnalysis:
        result = await self.llm.astructured_predict(
            output_cls=QueryAnalysis,
            prompt=self.prompt,
            query=query,
        )
        print(result)
        return result


async def main():
    await analyzer.analyze("帮我找找TileGS的方法是什么")
    
if __name__ == '__main__':
    ollama = OllamaServer()
    ollama.start_all()

    analyzer = QueryAnalyzer(ollama.create_ollama_llm("LLM"))
    # analyzer = QueryAnalyzer()

    asyncio.run(main())

    ollama.stop_all()