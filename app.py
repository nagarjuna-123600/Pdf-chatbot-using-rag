import os
import pdfplumber
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
os.environ['GROQ_API_KEY'] = groq_key
groq_api_model_name = os.getenv("GROQ_API_MODEL_NAME")
st.header("MY FIRST CHATBOT")
huggingface_embeddings_model_name = os.getenv("HUGGINGFACE_EMBEDDINGS_MODEL_NAME")

with st.sidebar:
    st.title("Your Documents")
    uploaded_pdf_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )






# extract contents from files PDF and check it
if uploaded_pdf_files:

    # 1. Initialize chat history (TOP)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    text = ""
    for uploaded_pdf_file in uploaded_pdf_files:
        with pdfplumber.open(uploaded_pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    # st.write(text)

    # Splitting into chuncks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

    # Generating embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=huggingface_embeddings_model_name
    )

    # store embeddings in vector db
    vector_store = FAISS.from_texts(chunks, embeddings)

    #get user question
    user_question = st.chat_input("Type your question here")

    #generate answer
    #(CHAIN) question -> embeddings -> similarity search -> results to LLM -> response

    def format_docs(docs):
         return "\n\n".join([doc.page_content for doc in docs])


    retriever = vector_store.as_retriever(
         search_type="mmr",
         search_kwargs={"k":4}
     )

    llm = ChatGroq(
         model_name=groq_api_model_name,
         temperature=0.3

     )

    prompt = ChatPromptTemplate.from_messages(
        [
             ("system",
                 "You are a helpful assistant answering questions about a PDF document.\n\n"
                         "Use ONLY  context provided below to answer.\n\n"
               "Context:\n{context}\n\n"
               "Please remind this chat history and use for entire conversation Chat History:\n{chat_history}\n\n"
               "You MUST use chat history to understand follow-up questions.\n\n"
               "If the question refers to something earlier, use chat history.\n\n"
               "Guidelines:\n"
               "1. Include relevant details, brief answer\n"
               "2. Output in bullet points\n"
               "3. Always start with heading 'YOUR ANSWER :'\n"
             ),
             ("human", "{question}")

        ]
    )

    #adding previous memory to llm
    def format_chat_history(chat_history):
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            print("***********no chat history**********************")
        return "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in chat_history
        ])



    chain = (
        {"context":retriever | format_docs,
         "question": RunnablePassthrough(),
         "chat_history": lambda x: format_chat_history(
            st.session_state.get("chat_history", [])
        )}
        | prompt
        | llm
        | StrOutputParser()
    )

    # print(st.session_state.chat_history)


    if "chat_history" in st.session_state:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []

    # 3. Display previous chats
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_question:
        response = chain.invoke(user_question)

        with st.chat_message("user"):
            st.markdown(f"**Q:** {user_question}")

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.chat_history.append({
            "role": "user",
            "content": f"*{user_question}*"
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response
        })
