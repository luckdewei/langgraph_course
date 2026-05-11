from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch

load_dotenv()


@tool
def triple(num: float) -> float:
    """
    :param num: 需要乘以三的数字
    :return: num 的三倍
    """
    return 3 * float(num)


tools = [TavilySearch(max_results=1), triple]

llm = ChatDeepSeek(model="deepseek-chat").bind_tools(tools)
