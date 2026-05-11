import datetime

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek

from schemas import AnswerQuestion, ReviseAnswer

llm = ChatDeepSeek(model="deepseek-chat")
# JsonOutputToolsParser
# 用途：将模型返回的工具调用请求直接解析为 JSON 对象。
# 适用场景：当你只需要获取模型想要调用的工具名称及其参数，而不需要将其转换为复杂的类实例时。
# 输出格式：一个列表，每个元素包含 type（工具名）和 args（工具参数字典）。
# [{'type': 'search', 'args': {'query': 'langgraph'}}]
parser = JsonOutputToolsParser(return_id=True)
# PydanticToolsParser
# 用途：将模型返回的工具调用结果解析为 Pydantic 模型实例（对象）。
# 适用场景：这是处理工具调用最常用的方式。如果你已经定义了 Pydantic 类作为工具的入参 schema，使用此解析器可以直接拿到该类的实例。
# 优势：
# 类型安全：可以直接调用 instance.field_name 获取参数，IDE 有自动补全。
# 自动验证：Pydantic 会自动对解析出的参数进行类型校验，如果不符合你定义的 schema，会直接抛出错误。
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位专业研究员。
当前时间: {time}

1. {first_instruction}
2. 反思并批评你的答案。要严格以最大化改进。
3. 推荐搜索查询来研究信息并改进你的答案。""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "使用要求的格式回答用户的问题。"),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="提供详细的约250词的答案。"
)
# 强制模型调用特定工具 (tool_choice)
# 如果你希望模型总是调用某个特定工具，或者强制它调用至少一个工具，可以使用 tool_choice 参数。
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """使用新信息修改你之前的答案。
        - 你应该使用之前的批评来在你的答案中添加重要信息。
        - 你必须包含数字引用以确保你的修订答案可以被验证。
        - 在答案底部添加一个"参考文献"部分（不计入字数限制）。格式如下：
            - [1] https://example.com
            - [2] https://example.com
        - 你应该使用之前的批评来删除答案中多余的信息，并确保不超过250词。
"""
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


if __name__ == "__main__":
    human_message = HumanMessage(
        content="写一篇关于AI驱动的SOC/自主SOC问题领域的文章，"
        "列出从事该领域并获得融资的初创公司。"
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)
