from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings

# from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()

embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
persist_directory = "./.chroma"
collection_name = "rag-chroma"

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
# 数组扁平化
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name=collection_name,
    embedding=embeddings,
    persist_directory=persist_directory,
)

retriever = Chroma(
    collection_name=collection_name,
    persist_directory=persist_directory,
    embedding_function=embeddings,
).as_retriever()
