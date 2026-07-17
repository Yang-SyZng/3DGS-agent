from tavily import TavilyClient
from config.settings import setting
tavily_client = TavilyClient(api_key=setting.TAVILY_API_KEY)
response = tavily_client.search("AbsGS 完整标题")
print(response)