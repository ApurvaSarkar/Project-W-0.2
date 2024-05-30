from typing import List
from PIL import Image
import streamlit as st
from phi.assistant import Assistant
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.document.reader.website import WebsiteReader
from phi.utils.log import logger
from assistant import get_rag_assistant 


pageicon = Image.open("D:/code/college/Project/Project W 0.2/src/files/logo.png")
st.set_page_config(
    page_title="Project W",
    page_icon=pageicon,
    layout="wide",
)
st.image(pageicon, width=200)
st.title("Local Generative Chatbot RAG")

def restart_assistant():
    st.session_state["rag_assistant"] = None
    st.session_state["rag_assistant_run_id"] = None
    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1
    if "file_uploader_key" in st.session_state:
        st.session_state["file_uploader_key"] += 1
    st.rerun()

def main() -> None:
    llm_model = st.sidebar.selectbox("Select Model", options=["llama3","phi", "gemma", "mixtral","llama2-uncensored"])
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model
    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        restart_assistant()

    # Get Embeddings model
    embeddings_model = st.sidebar.selectbox(
        "Select Embeddings",
        options=["nomic-embed-text", "llama3", "llama2"],
    )
    # Set assistant_type in session state
    if "embeddings_model" not in st.session_state:
        st.session_state["embeddings_model"] = embeddings_model
    elif st.session_state["embeddings_model"] != embeddings_model:
        st.session_state["embeddings_model"] = embeddings_model
        st.session_state["embeddings_model_updated"] = True
        restart_assistant()

    # Get the assistant
    rag_assistant: Assistant
    if "rag_assistant" not in st.session_state or st.session_state["rag_assistant"] is None:
        logger.info(f"---*--- Creating {llm_model} Assistant ---*---")
        rag_assistant = get_rag_assistant(llm_model=llm_model, embeddings_model=embeddings_model)
        st.session_state["rag_assistant"] = rag_assistant
    else:
        rag_assistant = st.session_state["rag_assistant"]

    # Create assistant run (i.e. log to database) and save run_id in session state
    try:
        st.session_state["rag_assistant_run_id"] = rag_assistant.create_run()
    except Exception:
        st.warning("Could not create assistant, is the database running?")
        return

    # Load existing messages
    MAX_CHAT_HISTORY_LENGTH = 10

    assistant_chat_history = rag_assistant.memory.get_chat_history()
    if len(assistant_chat_history) > 0:
        logger.debug("Loading chat history")
        st.session_state["messages"] = assistant_chat_history[-MAX_CHAT_HISTORY_LENGTH:]
    else:
        logger.debug("No chat history found")
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi what would you like to know?"}]

    # Prompt for user input
    if prompt := st.chat_input():
        st.session_state["messages"].append({"role": "user", "content": prompt})

    # Display existing chat messages
    for message in st.session_state["messages"]:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is from a user, generate a new response
    last_message = st.session_state["messages"][-1]
    if last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            response = ""
            resp_container = st.empty()
            for delta in rag_assistant.run(question):
                response += delta
                resp_container.markdown(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})

    # Load knowledge base
    if rag_assistant.knowledge_base:
        if "url_scrape_key" not in st.session_state:
            st.session_state["url_scrape_key"] = 0

        input_url = st.sidebar.text_input(
            "Add URL to Knowledge Base", type="default", key=st.session_state["url_scrape_key"]
        )
        add_url_button = st.sidebar.button("Add URL")
        if add_url_button:
            if input_url is not None:
                alert = st.sidebar.info("Processing URLs...", icon="ℹ️")
                if f"{input_url}_scraped" not in st.session_state:
                    scraper = WebsiteReader(max_links=2, max_depth=1)
                    web_documents: List[Document] = scraper.read(input_url)
                    if web_documents:
                        rag_assistant.knowledge_base.load_documents(web_documents, upsert=True)
                    else:
                        st.sidebar.error("Could not read website")
                    st.session_state[f"{input_url}_uploaded"] = True
                alert.empty()

        # Add PDFs to knowledge base
        if "file_uploader_key" not in st.session_state:
            st.session_state["file_uploader_key"] = 100

        uploaded_file = st.sidebar.file_uploader(
            "Add a PDF :page_facing_up:", type="pdf", key=st.session_state["file_uploader_key"]
        )
        if uploaded_file is not None:
            alert = st.sidebar.info("Processing PDF...", icon="🧠")
            rag_name = uploaded_file.name.split(".")[0]
            if f"{rag_name}_uploaded" not in st.session_state:
                reader = PDFReader()
                rag_documents: List[Document] = reader.read(uploaded_file)
                if rag_documents:
                    rag_assistant.knowledge_base.load_documents(rag_documents, upsert=True)
                else:
                    st.sidebar.error("Could not read PDF")
                st.session_state[f"{rag_name}_uploaded"] = True
            alert.empty()

    if rag_assistant.knowledge_base and rag_assistant.knowledge_base.vector_db:
        if st.sidebar.button("Clear Knowledge Base"):
            rag_assistant.knowledge_base.vector_db.clear()
            st.sidebar.success("Knowledge base cleared")

    if st.sidebar.button("New Run"):
        restart_assistant()

main()
