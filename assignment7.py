import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_community.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.schema.runnable import RunnablePassthrough
from pathlib import Path
import json

class JsonOutputParser(BaseOutputParser):

    def parse(self, text):
        text = text.replace("```", "").replace("json", "")

        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="Assignment 7",
    page_icon="😍",
)

st.title("ASSIGNMENT 7")

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    # docs = retriever.get_relevant_documents(term)
    docs = retriever.invoke(term)
    return docs

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    Path("./.cache/quiz_files").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_chain, _docs, topic):
    # chain = {"context": questions_chain} | formatting_chain | output_parser
    # return chain.invoke(_docs)

    response = _chain.invoke(_docs)
    response = response.additional_kwargs["function_call"]["arguments"]

    quiz = output_parser.parse(response)
    # print(quiz)
    return quiz


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def get_difficulty(arg):
    print(difficulty)
    return difficulty

function_prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant that is role playing as a teacher.
Based ONLY on the following context make 10 (TEN) questions minimum to test
 the user's knowledge about the text.
 The user can select the exam difficulty level, 
 and you must ensure that this difficulty level is reflected 
 when creating the exam questions. 

 difficulty level: {difficulty}
 Context: {context}
"""
)

topic = None
file = None

with st.sidebar:
    openai_api_key = st.text_input(
        label="OPENAI API KEY",
    )

    if openai_api_key:
        
        difficulty = st.selectbox("Choose the exam difficulty level:", ["Easy", "Medium", "Hard",],)           
        
        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
        )

        if choice == "File":
            file = st.file_uploader(
                "Upload a .txt .pdf or .docx file",
                type=["pdf", "txt", "docx"],
            )
        else:
            topic = st.text_input("Search Wikipedia...")


        

def main():
    if not openai_api_key:
        return
    
    llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    ).bind(
        function_call="auto",
        functions=[
            function,
        ],
    )

    docs = None

    if file:
        docs = split_file(file)

    if topic:
        docs = wiki_search(topic)


    function_chain = {"difficulty":get_difficulty, "context": format_docs} | function_prompt | llm

    if not docs:
        st.markdown(
        """
Welcome to QuizGPT.

I will make a quiz from Wikipedia articles or files you upload to test your
 knowledge and help you study.

Get started by uploading a file or searching on Wikipedia in the sidebar.

"""
    )
    else:
        response = run_quiz_chain(function_chain, docs, topic if topic else file.name,)
        # st.write(response)

        success_count = 0

        with st.form("questions_form"):
            for idx, question in enumerate(response["questions"]):
                st.write(question["question"])

                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    key=idx,
                )


                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                    success_count += 1
                elif value is not None:
                    st.error("Wrong!")



            submitted = st.form_submit_button(disabled=success_count == 10)

            if success_count == 10:
                st.balloons()


try:
    main()
except Exception as e:
    st.error("Check your OpenAI API Key or File")
    st.write(e)