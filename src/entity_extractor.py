from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class Entities(BaseModel):
    names: list[str] = Field(..., description="List of entities")

entity_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting organization and person entities from the text."),
    ("human", "Use the given format to extract information from the following input: {question}")
])

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
entity_chain = entity_prompt | llm.with_structured_output(Entities)

def extract_entities(question: str) -> list[str]:
    entities = entity_chain.invoke({"question": question})
    return entities.names
