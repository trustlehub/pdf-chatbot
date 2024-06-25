import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent

import os

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


embeddings = SpacyEmbeddings(model_name="en_core_web_sm")


def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


def get_conversational_chain(tools, ques, chat_history):
    os.environ["ANTHROPIC_API_KEY"] = os.getenv["ANTHROPIC_API_KEY"]
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,
                     api_key=os.environ["ANTHROPIC_API_KEY"])

    # Prepare the chat history messages for the prompt
    prompt_messages = [
        ("system",
         "You are a helpful assistant. Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, just say, 'answer is not available in the context'. Don't provide the wrong answer."),
    ]

    for message in chat_history:
        if message['role'] == 'human':
            prompt_messages.append(("human", message['content']))
        elif message['role'] == 'assistant':
            prompt_messages.append(("assistant", message['content']))

    prompt_messages.append(("human", ques))

    prompt = ChatPromptTemplate.from_messages(
        prompt_messages + [("placeholder", "{agent_scratchpad}")]
    )

    tool = [tools]
    agent = create_tool_calling_agent(llm, tool, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    response = agent_executor.invoke({"input": ques, "chat_history": chat_history})
    return response['output']


def user_input(user_question, chat_history):
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    retrieval_chain = create_retriever_tool(retriever, "pdf_extractor",
                                            "This tool is to give answer to queries from the pdf")
    response = get_conversational_chain(retrieval_chain, user_question, chat_history)
    return response


def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Interactive Chat with PDF")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                   accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if st.button("Send"):
        if user_question:
            response = user_input(user_question, st.session_state.chat_history)
            st.session_state.chat_history.append(
                {"role": "human", "content": user_question})
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})

    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            if chat['role'] == 'human':
                st.write(f"**You:** {chat['content']}")
            elif chat['role'] == 'assistant':
                st.write(f"**Assistant:** {chat['content']}")


if __name__ == "__main__":
    main()
