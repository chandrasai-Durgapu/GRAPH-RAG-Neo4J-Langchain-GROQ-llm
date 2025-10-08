from src.wikipedia_loader import load_wikipedia_docs
from data_ingestion.graph_ingestion import ingest_documents_to_neo4j
from src.neo4j_client import Neo4jClient
from src.qa_chain import answer_question

def main():
    print("Loading Wikipedia documents...")
    docs = load_wikipedia_docs("Elizabeth I")
    print(f"Loaded {len(docs)} chunks")

    print("Ingesting documents into Neo4j...")
    ingest_documents_to_neo4j(docs)

    print("Running example questions:")
    question1 = "Which house did Elizabeth I belong to?"
    answer1 = answer_question(question1)
    print(f"Q: {question1}\nA: {answer1}\n")

    question2 = "When was she born?"
    answer2 = answer_question(question2, chat_history=[(question1, answer1)])
    print(f"Q: {question2}\nA: {answer2}")

if __name__ == "__main__":
    main()
