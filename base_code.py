from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

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
    embedding=FakeEmbeddings(size=1352),
    persist_directory="./test_db"
)

query = "?"
result = vector_store.search_similarity(query)
for i, document in result:
    print(f"{i+1}번쨰 문서 : {document.page_content}")
    print(f"출처 : {document.metadata}")
