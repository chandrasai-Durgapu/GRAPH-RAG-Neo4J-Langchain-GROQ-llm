from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from logger.logger import get_logger

logger = get_logger("Wikipedia Content")

def load_wikipedia_docs(query: str, chunk_size=512, chunk_overlap=25, max_docs=3):
    """
    Load and split Wikipedia documents for a given query.

    Args:
        query (str): Search term to fetch Wikipedia articles.
        chunk_size (int): Maximum tokens per chunk.
        chunk_overlap (int): Tokens to overlap between chunks.
        max_docs (int): Maximum number of articles to load.

    Returns:
        list: List of split document chunks.
    """
    try:
        logger.info(f"Downloading Wikipedia data for query: '{query}'")
        raw_docs = WikipediaLoader(query=query).load()
        docs = raw_docs[:max_docs]

        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(docs)

        logger.info(f"Loaded {len(docs)} docs and split into {len(split_docs)} chunks")
        return split_docs

    except Exception as e:
        logger.error(f"Error loading Wikipedia docs: {e}", exc_info=True)
        return []
