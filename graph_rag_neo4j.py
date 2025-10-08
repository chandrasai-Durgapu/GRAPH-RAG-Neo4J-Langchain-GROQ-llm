import os
import re
from typing import List, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network
from IPython.display import IFrame  # Only if using Jupyter; else remove

from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import to avoid deprecation
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Neo4j credentials from .env
NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Make sure this is set for ChatGroq


# === Connect to Neo4j ===
graph = Neo4jGraph(
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)


# === Load Wikipedia Data ===
def load_wikipedia_docs(query: str, max_pages: int = 3):
    loader = WikipediaLoader(query=query)
    raw_docs = loader.load()
    return raw_docs[:max_pages]


# === Split documents ===
def split_documents(documents, chunk_size=512, chunk_overlap=25):
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


# === Initialize LLM ===
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Update with your Groq model name
    temperature=0,
)


# === Graph ingestion ===
def ingest_graph(documents):
    from langchain_experimental.graph_transformers import LLMGraphTransformer

    transformer = LLMGraphTransformer(llm=llm)
    graph_documents = transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(
        graph_documents=graph_documents,
        baseEntityLabel=True,
        include_source=True,
    )
    print(f"Ingested {len(graph_documents)} graph documents")


# === Vector Index for Neo4j ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_index = Neo4jVector.from_existing_graph(
    embedding=embeddings,
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)


# === Entity extraction model ===
class Entities(BaseModel):
    names: List[str] = Field(..., description="Person, organization, or business entities in text")


entity_extraction_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are extracting organization and person entities from the text."),
        ("human", "Extract entities from the following input: {question}"),
    ]
)

entity_chain = entity_extraction_prompt | llm.with_structured_output(Entities)


# === Helper to clean Lucene query strings for Neo4j fulltext search ===
def remove_lucene_chars(text: str) -> str:
    return re.sub(r'[-+!(){}[\]^"~*?:\\/]|&&|\|\|', ' ', text)


def generate_full_text_query(input_str: str) -> str:
    cleaned = remove_lucene_chars(input_str)
    words = [w.strip() for w in cleaned.split() if w.strip()]
    if not words:
        return ""
    return " AND ".join(f"{word}~2" for word in words)


# === Structured retriever querying Neo4j fulltext ===
def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})

    for entity in entities.names:
        fulltext_query = generate_full_text_query(entity)

        cypher = """
        CALL db.index.fulltext.queryNodes('entity', $query, {limit: 2})
        YIELD node, score
        
        CALL {
            WITH node
            MATCH (node)-[r]->(neighbor)
            WHERE type(r) <> 'MENTIONS'
            RETURN node.id + ' -[' + type(r) + ']-> ' + neighbor.id AS output
            UNION ALL
            WITH node
            MATCH (node)<-[r]-(neighbor)
            WHERE type(r) <> 'MENTIONS'
            RETURN neighbor.id + ' <-[' + type(r) + ']- ' + node.id AS output
        }
        
        RETURN output
        LIMIT 50
        """

        try:
            response = graph.query(cypher, {"query": fulltext_query})
            result += f"\n🔎 Entity: {entity}\n"
            result += "\n".join(el["output"] for el in response) + "\n"
        except Exception as e:
            result += f"\n⚠️ Error querying entity '{entity}': {e}\n"

    return result.strip()


# === Combined retriever for structured + vector search ===
def retriever(question: str) -> str:
    print(f"Running retriever for question: {question}")
    structured_data = structured_retriever(question)
    unstructured_docs = vector_index.similarity_search(question, k=5)
    unstructured_data = [doc.page_content for doc in unstructured_docs]

    combined = f"""Structured data:
{structured_data}

Unstructured data:
{"\n# Document: ".join(unstructured_data)}
"""
    return combined


# === Chat history formatter ===
def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List[str]:
    buffer = []
    for human, ai in chat_history:
        buffer.append(f"Human: {human}")
        buffer.append(f"AI: {ai}")
    return buffer


# === Prompt to condense follow-up questions ===
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that rewrites follow-up questions into standalone questions.",
        ),
        (
            "human",
            "{chat_history}\nFollow-up question: {question}",
        ),
    ]
)


# === RunnableBranch to handle chat history condensing ===
_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    RunnableLambda(lambda x: x["question"]),
)


# === Final QA prompt template ===
QA_PROMPT_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""

prompt = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)


# === Final QA chain combining retriever and LLM ===
chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)


# === Example usage ===
if __name__ == "__main__":
    # 1. Load & ingest Wikipedia docs (run once)
    print("Loading Wikipedia documents...")
    docs = load_wikipedia_docs("Elizabeth I")
    print(f"Loaded {len(docs)} documents")
    split_docs = split_documents(docs)
    print(f"Split into {len(split_docs)} chunks")
    print("Ingesting graph data to Neo4j...")
    ingest_graph(split_docs)

    # 2. Simple query without chat history
    question1 = "Which house did Elizabeth I belong to?"
    print("\nQuestion 1:", question1)
    answer1 = chain.invoke({"question": question1})
    print("Answer 1:", answer1)

    # 3. Follow-up question with chat history
    question2 = "When was she born?"
    chat_history = [(question1, answer1)]
    print("\nQuestion 2 (follow-up):", question2)
    answer2 = chain.invoke({"question": question2, "chat_history": chat_history})
    print("Answer 2:", answer2)
