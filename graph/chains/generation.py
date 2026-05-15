from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
prompt = Client().pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)

generation_chain = prompt | llm | StrOutputParser()
