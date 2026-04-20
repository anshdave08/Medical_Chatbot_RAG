import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from utils.safety import classify_query, violence_response, self_harm_response, emotional_response
import os

from utils.symptom_matcher import match_disease, get_disease_info
from utils.query_rewrite import rewrite_query

def is_small_talk(text):
    text = text.lower().strip()
    small_talk = ["thank you", "thanks", "ok", "okay", "nice", "good", "great", "bye"]
    return any(word in text for word in small_talk)

load_dotenv()

# ---------------- UI ----------------
st.set_page_config(page_title="Medical Chatbot", layout="wide")
st.title("🩺 Medical Chatbot cum Family Doctor")

# ---------------- Chat History ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Memory (persistent) ----------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

memory = st.session_state.memory

# ---------------- Load Embeddings ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- Load Vector DB ----------------
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10}
)
# ---------------- Groq LLM ----------------
llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ---------------- RAG Chain ----------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever,
    memory=memory
)

# ---------------- Show Previous Messages ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- Chat Input ----------------
query = st.chat_input("Ask your medical question...")

if query:

    # ---------------- 1️⃣ CLASSIFY QUERY ----------------
    category = classify_query(query)

    if category == "violence":
        answer = violence_response()

    elif category == "self_harm":
        answer = self_harm_response()

    elif category == "emotional":
        answer = emotional_response()

    else:
        answer = None

    # ---------------- 🚨 STOP IF UNSAFE ----------------
    if answer:
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        st.stop()


    # ---------------- 2️⃣ SMALL TALK ----------------
    if is_small_talk(query):
        response_text = "You're welcome 😊. Let me know if you have any medical questions."

        with st.chat_message("assistant"):
            st.markdown(response_text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text
        })

        st.stop()


    # ---------------- 3️⃣ SHOW USER MESSAGE ----------------
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)


    # ---------------- 4️⃣ JSON DISEASE MATCH ----------------
    disease_info = get_disease_info(query)

    if disease_info:
        answer = f"""
### {disease_info['disease']}

Symptoms:
{', '.join(disease_info['symptoms'])}

Treatment:
{disease_info['treatment']}

⚠ This is not a medical diagnosis.
"""

    else:
        # ---------------- 5️⃣ LLM + SYMPTOM MATCH ----------------
        predicted = match_disease(query)

        if predicted:
            disease_list = [d["disease"] for d in predicted]

            response = qa_chain.invoke({
                "question": f"""
You are an experienced medical assistant.

User Input:
{query}

Possible Conditions:
{disease_list}

Provide structured response:

1. Possible Conditions
2. Reasoning
3. Treatment
4. Precautions
5. When to see a doctor

Do NOT hallucinate.
Be confident and precise.

⚠ This is not a medical diagnosis.
""",
                "chat_history": memory.chat_memory.messages
            })

        else:
            response = qa_chain.invoke({
                "question": f"""
You are a medical assistant.

If the answer is NOT found in medical context, respond strictly:
"I am not confident about this. Please consult a medical professional."

User query:
{query}
""",
                "chat_history": memory.chat_memory.messages
            })

        answer = response["answer"]

        # ---------------- 6️⃣ HALLUCINATION GUARD ----------------
        if len(answer.strip()) < 20 or "not mentioned" in answer.lower():
            answer = "⚠ I am not confident about this. Please consult a doctor."


    # ---------------- 7️⃣ SHOW RESPONSE ----------------
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })