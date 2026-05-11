from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessagesState, StateGraph

from nodes import run_agent_reasoning_engine, tool_node

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: dict) -> str:
    # 当 state["messages"][-1] 不在有 tool_calls 时，说明 Agent 认为它已经得到答案
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT


flow = StateGraph(MessagesState)

# add_node 接收两个主要参数：
# name（节点名称）：字符串，作为图内部引用该节点的 ID。
# # action（节点逻辑）：即该节点执行的函数、工具或子图对象。

flow.add_node(AGENT_REASON, run_agent_reasoning_engine)
# 设置 graph 入口
flow.set_entry_point(AGENT_REASON)

flow.add_node(ACT, tool_node)

# 设置条件边
flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {
        END: END,
        ACT: ACT,
    },
)

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    print("Hello ReAct with LangGraph")
    res = app.invoke(
        {"messages": [HumanMessage(content="上海的天气如何？列出它，然后将其乘以三")]}
    )
    print(res["messages"][LAST].content)
