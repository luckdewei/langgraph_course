from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """将用户问题路由到最相关的数据源。"""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="根据用户问题选择将其路由到网络搜索或向量存储。",
    )


llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """你是一个专家，擅长将用户问题路由到向量存储或网络搜索。
向量存储包含与 agents、提示工程和对抗攻击相关的文档。
对于这些主题的问题使用向量存储。其他所有问题使用网络搜索。"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
