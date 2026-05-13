from typing import List, TypedDict


#  LangGraph 要求: StateGraph 需要状态是 TypedDict 或 dict
class GraphState(TypedDict):
    """
    图的状态

    Attributes:
        question: 问题
        generation: LLM 生成
        web_search: 是否需要搜索
        documents: 文档列表
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
