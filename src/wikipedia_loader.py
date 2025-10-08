from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter

def load_wikipedia_docs(query: str, chunk_size=512, chunk_overlap=25, max_docs=3):
    raw_docs = WikipediaLoader(query=query).load()
    docs = raw_docs[:max_docs]

    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    return split_docs
