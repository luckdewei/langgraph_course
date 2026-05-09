from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位著名的Twitter网红，正在对一条推文进行评分。请为用户的推文提供评论和建议。"
            "始终提供详细的建议，包括对长度、传播力、风格等方面的要求。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名Twitter技术影响者助理，负责撰写优质的Twitter帖子。"
            "根据用户请求，生成尽可能优质的推文。"
            "如果用户提供了批评意见，请用你之前尝试过的修订版进行回应。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


llm = ChatDeepSeek(model="deepseek-v4-flash")
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
