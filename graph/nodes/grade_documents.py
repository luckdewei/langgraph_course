from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    判断检索到的文档是否与问题相关
    如果有任何文档不相关,将设置标志位以执行网络搜索

    Args:
        state (dict): 当前图状态

    Returns:
        state (dict): 过滤掉无关文档并更新 web_search 状态
    """
    print("---检查文档与问题的相关性---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: 文档相关---")
            filtered_docs.append(d)
        else:
            print("---GRADE: 文档不相关---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
