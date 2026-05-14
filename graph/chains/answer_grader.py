from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field


class GradeAnswer(BaseModel):

    binary_score: bool = Field(description="答案是否解决了问题，'yes' 或 'no'")


llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """
    你是一个评分员，评估答案是否解决了问题。
 
    给出一个二分制评分 'yes' 或 'no'。'Yes' 表示答案解决了问题。
"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "用户问题：\n\n {question} \n\n LLM 生成内容：{generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
