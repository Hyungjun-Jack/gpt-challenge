import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

st.set_page_config(
    page_title="Assignment 6",
    page_icon="😍",
)

st.title("ASSIGNMENT 6")


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(
        self,
        *args,
        **kwargs,
    ):
        self.message_box = st.empty()

    def on_llm_end(
        self,
        *args,
        **kwargs,
    ):
        save_message(self.message, "ai")

    def on_llm_new_token(
        self,
        token,
        *args,
        **kwargs,
    ):
        self.message += token
        self.message_box.markdown(self.message)


def save_message(message, role):
    st.session_state["messages"].append(
        {
            "message": message,
            "role": role,
        }
    )


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def print_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def load_memory(_):
    return memory.load_memory_variables({})["history"]


@st.cache_resource()
def load_session_state(_):
    return st.session_state["history_message"]


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_resource()
def make_llm(api_key):
    llm = ChatOpenAI(
        api_key=api_key,
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )

    memory = ConversationBufferMemory(
        llm=llm,
        return_messages=True,
        max_token_limit=120,
    )

    return {llm, memory}


@st.cache_resource(
    show_spinner="Embedding file...",
)
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    vectorstore = FAISS.from_documents(
        docs,
        cached_embeddings,
    )
    retriever = vectorstore.as_retriever()
    return retriever


file = None
api_key = None

with st.sidebar:
    api_key = st.text_input(
        label="OPENAI API KEY",
    )

    if api_key:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )
    else:

        st.session_state["messages"] = []
        st.session_state["history_message"] = []

if api_key:

    llm = ChatOpenAI(
        api_key=api_key,
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )

    memory = ConversationBufferMemory(
        llm=llm,
        return_messages=True,
        max_token_limit=120,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 아주 도움이되는 비서입니다. 당신은 다음의 context만 사용해서 대답해야합니다.
                만약 당신이 답을 모른다면 절대로 추측하지 말고 반드시 모른다고 얘기하세요.

                \n\n{context}""",
            ),
            MessagesPlaceholder(
                variable_name="history",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    st.markdown(
        """
Welcome!

Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
    """
    )
else:
    st.markdown(
        """
OPENAI API키를 입력하세요.
    """
    )


if file:
    retriever = embed_file(file)

    send_message("I'm ready! Ask away!", "ai", save=False)

    print_history()

    message = st.chat_input("Ask anything about your file...")

    if message:

        load_session_state({})

        send_message(message, "human")

        chain = (
            (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": Passthrough(),
                    "history": load_session_state,
                }
            )
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            response = chain.invoke(message)

        memory.save_context(
            {"input": message},
            {"output": response.content},
        )

        st.session_state["history_message"] += load_memory({})
