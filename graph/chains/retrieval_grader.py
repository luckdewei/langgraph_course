from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)


class GradeDocuments(BaseModel):
    """检索文档相关性检查"""

    binary_score: str = Field(description="文档是否与问题相关,'yes'或'no'")


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """你是一评估助手,用于评估检索到的文档与用户问题的相关性。\n
    如果文档包含与问题相关的关键词或语义意义,请将其评为相关。\n
    请给出'yes'或'no',表示文档是否与问题相关。"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "检索到的文档: \n\n {document} \n\n 用户问题: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
