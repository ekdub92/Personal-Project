from langchain_chroma import Chroma
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter 

# HuggingFace : opendatalab/ScienceMetaBench
json_data = JSONLoader(
    file_path="ebook_1022.jsonl",
    jq_schema=".",
    text_content=False,
    json_lines=True
).load()

splitter = RecursiveJsonSplitter(max_chunk_size=300)

vector_store = Chroma.from_documents(
    documents=json_data,
    embedding=FakeEmbeddings(size=1352),
    persist_directory="./test_db"
)

query = "?"
result = vector_store.similarity_search(query)
for i, document in enumerate(result):
    print(f"{i+1}번째 문서 : {document.page_content}")
    print(f"출처 : {document.metadata}")
