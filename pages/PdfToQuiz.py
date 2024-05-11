import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template
import openai

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            pass
        else:
            st.write(message.content)

def on_multiple_button_click():
    user_question = " Create multiple-choice questions and answers, and translate korean, only prints korean. "
    handle_userinput(user_question)


def on_short_answer_button_click():
    user_question = "Create short questions and answers, and translate korean, only prints korean."
    handle_userinput(user_question)


def on_true_false_button_click():
    user_question = "Create true of false questions and answers, and translate korean, only prints korean. "
    handle_userinput(user_question)


def on_blanks_button_click():
    user_question = "Create fill-in-the-blank questions and answers, and translate korean, only prints korean. The blanks should represent missing words or phrases in the questions, to be filled in by the reader."
    handle_userinput(user_question)


def on_multiple_button_click_eng():
    user_question = " Generate multiple-choice questions based on the given concept."
    handle_userinput(user_question)


def on_short_answer_button_click_eng():
    user_question = "Generate short-answer questions based on the given concept."
    handle_userinput(user_question)


def on_true_false_button_click_eng():
    user_question = "Create true or false questions and answers based on the given concept. "
    handle_userinput(user_question)


def on_blanks_button_click_eng():
    user_question = "Generate blanks questions based on the given concept."
    handle_userinput(user_question)


def main():

    load_dotenv()
    st.set_page_config(page_title="PDF 기반 문제 생성",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("퀴즈 생성기 :books:")
    st.caption("파일 업로드 후 원하시는 문제를 선택하여 주십시오. ")

    lang = st.radio(
        "언어 선택",
        ["영어","한국어"],
    )
    type = st.radio(
        "종류 선택",
        ["객관식", "주관식", "참/거짓", "빈칸 맞추기"],
    )

    if lang == '한국어':
        if type == '객관식':
            if st.button('생성'):
                st.session_state.chat_history = None
                on_multiple_button_click()

        if type == '주관식':
            if st.button('생성'):
                st.session_state.chat_history = None
                on_short_answer_button_click()

        if type == '참/거짓':
            if st.button('생성'):
                st.session_state.chat_history = None
                on_true_false_button_click()

        if type == '빈칸 맞추기':
            if st.button('생성'):
                st.session_state.chat_history = None
                on_blanks_button_click()

    if lang == '영어':
        if type == '객관식':
            if st.button('생성'):
                st.session_state.chat_history = None
                on_multiple_button_click_eng()

        if type == '주관식':
            if st.button('생성'):
                st.session_state.chat_history = None
                on_short_answer_button_click_eng()

        if type == '참/거짓':
            if st.button('생성'):
                st.session_state.chat_history = None
                on_true_false_button_click_eng()

        if type == '빈칸 맞추기':
            if st.button('생성'):
                st.session_state.chat_history = None
                on_blanks_button_click_eng()

    with st.sidebar:
        st.subheader("문서 업로드")
        pdf_docs = st.file_uploader(
            "PDF 문서 여러개 업로드 가능.", accept_multiple_files=True, type=["pdf"])
        if st.button("벡터 변환"):
            with st.spinner("변환 중"):
                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)

                st.success('저장 완료!', icon="✅")


if __name__ == '__main__':
    main()
