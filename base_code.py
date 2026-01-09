from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
documents = [
    Document(
        page_content="?",
        metadata={"category": "?"}
    ),
    Document(
        page_content="!",
        metadata={"category": "!"}
    )
]

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./test_db"
)

query = "?"
result = vector_store.search_similarity(query)
for i, document in result:
    print("{i+1}번쨰 문서 : {document.page_content}")
    print("출처 : {document.meta_source}")
