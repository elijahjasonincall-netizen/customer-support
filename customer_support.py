from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st

# -------load & chunk----------------
loader = TextLoader("docs.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

# -------embeddings + chroma----------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)

# -------retriever----------------
retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}
)

# -------model----------------
model = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"], 
    model_name="llama-3.1-8b-instant",
)
print("✅ Groq LLM ready!")

# -------memory----------------
chat_history = []


# -------answer function----------------
def answer_question(question: str) -> str:
    # Get relevant chunks
    relevant_chunks = retriever.invoke(question)
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    # Debug: see retrieved context
    # Build messages list to send to model
    messages = [
        # 1. System message - tells bot who it is
        SystemMessage(
            content=f"""You are a helpful customer support assistant.
Use ONLY the following context to answer the customer's question.
Be concise and direct in your answers.
Do NOT include specific customer cases or examples.
If you don't know the answer, say "I'm sorry, I don't have information about that." and don't provide any further speculation.
Pay close attention to:
- inStock numbers (if inStock > 0, item IS available)
- eta means delivery days
- price is in the local currency
Be precise with numbers and stock levels.

Context: {context}"""
        ),
        # 2. All previous conversation ✅ Bot reads this every time!
        *chat_history,
        # 3. New question from customer
        HumanMessage(content=question),
    ]

    # Send everything to model
    response = model.invoke(messages)

    # ✅ Save to memory so next question has context
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))

    return response.content


# -------streamlit interface----------------
st.title("🤖 Customer Support Chatbot")

question = st.text_input("Ask a question:")

if st.button("Send"):
    if question:
        with st.status("🤖 Thinking...", expanded=True) as status:
            response = answer_question(question)

            status.update(label="✅ Done!", state="complete")

        st.write(f"**Bot:** {response}")
    else:
        st.warning("Please enter a question!")
