from typing import List

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="对缺失内容的批评。")
    superfluous: str = Field(description="对多余内容的批评。")


class AnswerQuestion(BaseModel):
    """回答问题。"""

    answer: str = Field(description="~250词的详细答案。")
    reflection: Reflection = Field(description="对初始答案的反思。")
    search_queries: List[str] = Field(
        description="1-3个搜索查询，用于研究改进当前答案的方法。"
    )


class ReviseAnswer(AnswerQuestion):
    """修改你的原始答案。"""

    references: List[str] = Field(description="支撑你更新答案的引用。")
