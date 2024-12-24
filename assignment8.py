import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SitemapLoader
from langchain_community.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document


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
        # print(self.message)
        pass

    def on_llm_new_token(
        self,
        token,
        *args,
        **kwargs,
    ):
        self.message += token
        self.message_box.markdown(self.message)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "history" not in st.session_state:
    st.session_state["history"] = []


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


st.set_page_config(
    page_title="Assignment 8",
    page_icon="😍",
)

st.title("ASSIGNMENT 8")


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()

    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
        filter_urls=[
            r"^(.*\/(?:ai-gateway|vectorize|workers-ai)\/).*",
        ],
    )

    loader.requests_per_second = 3

    docs = loader.load_and_split(text_splitter=splitter)

    vector_store = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(
            openai_api_key=openai_api_key,
        ),
    )

    return vector_store.as_retriever()


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't
    just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high,
     else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer
             the user's question.
            Use the answers that have the highest score (more helpful)
             and favor the most recent ones.
            Cite sources and return the sources of the answers as they are,
             do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm

    condensed = "\n\n".join(
        f"""{answer['answer']}\n
        Source:{answer['source']}\n
        Date:{answer['date']}\n"""
        for answer in answers
    )

    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def find_history(query):
    histories = st.session_state["history"]

    if len(histories) == 0:
        return None

    docs = [
        Document(
            page_content=f"""input: {history['question']}\n\n\n
            output: {history['answer']}"""
        )
        for history in histories
    ]

    print(docs)
    print("\n\n\n\n")

    vector_store = FAISS.from_documents(
        docs,
        OpenAIEmbeddings(openai_api_key=openai_api_key),
    )
    found_docs = vector_store.similarity_search_with_relevance_scores(
        query,
        k=1,
        score_threshold=0.7,
    )

    if not found_docs:
        return None

    doc, score = (
        found_docs[0][0],
        found_docs[0][1],
    )

    print(doc, score)

    found = doc.page_content.split("\n\n\n")[1].replace("output: ", "")

    return found


st.markdown(
    """
    Ask questions about the content of a website.

    Start by writing the OPENAI API KEY and URL of the website on the sidebar.
"""
)

with st.sidebar:
    openai_api_key = st.text_input(
        label="OPENAI API KEY",
    )

    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com/sitemap.xml",
    )

    st.markdown("---")
    st.write("Github: https://github.com/Hyungjun-Jack/assignment8")

if openai_api_key:
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )

if url:
    if not openai_api_key:
        st.error("OPENAI API키를 입력하세요.")
    else:
        if ".xml" not in url:
            st.error("Please write down a sitemap url.")
        else:
            retriever = load_website(url)

            print_history()

            query = st.chat_input("Ask a question to the website.")

            if query:
                send_message(query, "human")

                found = find_history(query)

                if found:
                    send_message(found, "ai")
                else:

                    chain = (
                        {
                            "docs": retriever,
                            "question": RunnablePassthrough(),
                        }
                        | RunnableLambda(get_answers)
                        | RunnableLambda(choose_answer)
                    )

                    placeholder = st.empty()

                    with placeholder.container():
                        with st.chat_message("ai"):
                            result = chain.invoke(query)
                            result_content = result.content.replace("$", r"\$")
                        placeholder.empty()

                    st.session_state["history"].append(
                        {
                            "question": query,
                            "answer": result_content,
                        }
                    )
                    send_message(
                        result_content,
                        "ai",
                    )
