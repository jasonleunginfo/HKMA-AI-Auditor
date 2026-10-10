import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

# OpenAI Key here
os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_OPENAI_KEY_HERE"

# Setup OpenAI Brain 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
vectorstore = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

def logician_agent(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = ChatPromptTemplate.from_template("""
    You are a Formal Logic Verifier for HKMA Banking Regulations.
    CONTEXT: {context}
    QUESTION: {question}
    TASK: Answer using ONLY the context. If unsure, say 'Insufficient evidence.'
    """)
    
    chain = prompt | llm
    return chain.invoke({"context": context, "question": question}).content

if __name__ == "__main__":
    query = "What are the board's responsibilities for technology risk?"
    print(f"Response: {logician_agent(query)}")
