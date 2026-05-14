from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)


class GradeHallucinations(BaseModel):
    """生成答案中是否存在幻觉的二分制评分"""

    binary_score: bool = Field(description="答案是否基于事实，'yes' 或 'no'")


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """你是一个评分员，评估 LLM 生成内容是否基于/支持一组检索到的事实。
 
    给出一个二分制评分 'yes' 或 'no'。'Yes' 表示答案基于/支持这组事实。"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "事实集：\n\n {documents} \n\n LLM 生成内容：{generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
