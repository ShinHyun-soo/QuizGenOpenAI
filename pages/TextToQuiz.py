import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

from typing import List
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field

import numpy as np
import sys

sys.path.append('../')  # 상위 폴더를 시스템 경로에 추가
from SubToQuiz import QuizMultipleChoice, QuizTrueFalse, QuizOpenEnded, create_quiz_chain, \
    create_multiple_choice_template, create_true_false_template, create_open_ended_template
from htmlTemplates import css, bot_template


# PDF 텍스트 추출

# 텍스트를 청크로 분할
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# 벡터스토어 생성
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# 청크 선택 알고리즘
def select_chunk_set(vectorstore, text_chunks, num_vectors=5):
    # 구현의 편의를 위해 앞쪽 5개의 텍스트 가져옴
    chunks = text_chunks[:5]
    # 선택된 텍스트 청크들을 하나의 문자열로 결합
    context = ' '.join(chunks)
    return context


def main():
    st.set_page_config(page_title="Text 기반 문제 생성",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("퀴즈 생성기 :books:")
    st.caption("Text 업로드 후 원하시는 문제를 선택하여 주십시오. ")

    language = st.radio(
        "언어 선택",
        ["English", "한국어", "Spanish", "French"],
    )
    quiz_type = st.radio(
        "종류 선택",
        ["multiple-choice", "true-false", "open-ended"],
    )
    num_questions = st.number_input("Enter the number of questions", min_value=1, max_value=10, value=3)
    llm = ChatOpenAI(model="gpt-4-turbo")

    if "context" not in st.session_state:
        st.session_state.context = None  # context 초기화

    with st.sidebar:
        st.subheader("텍스트 입력")
        raw_text = st.text_area("텍스트를 입력하세요.", height=200)
        if st.button("벡터 변환"):
            with st.spinner("변환 중"):

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.context = select_chunk_set(vectorstore, text_chunks)

                st.success('저장 완료!', icon="✅")

    # 퀴즈 유형 변경 시 상태 초기화
    if 'quiz_type' not in st.session_state or st.session_state.quiz_type != quiz_type:
        st.session_state.quiz_type = quiz_type
        st.session_state.quiz_data = None
        st.session_state.user_answers = None

    if st.button("Generate Quiz"):
        if st.session_state.context:
            if quiz_type == "multiple-choice":
                prompt_template = create_multiple_choice_template(language)
                pydantic_object_schema = QuizMultipleChoice
            elif quiz_type == "true-false":
                prompt_template = create_true_false_template(language)
                pydantic_object_schema = QuizTrueFalse
            else:
                prompt_template = create_open_ended_template(language)
                pydantic_object_schema = QuizOpenEnded

            st.write("에러가 발생할 경우, 다시 생성버튼을 눌러주시면 됩니다 ㅎㅎ")
            chain = create_quiz_chain(prompt_template, llm, pydantic_object_schema)
            st.session_state.quiz_data = chain.invoke(
                {"num_questions": num_questions, "quiz_context": st.session_state.context})
            st.session_state.user_answers = [None] * len(
                st.session_state.quiz_data.questions) if st.session_state.quiz_data else []
        else:
            st.write("텍스트를 왼쪽 슬라이드에 올리고 벡터 변환해주세요")

    if 'quiz_data' in st.session_state and st.session_state.quiz_data:
        user_answers = {}
        for idx, question in enumerate(st.session_state.quiz_data.questions):
            st.write(f"**{idx + 1}. {question}**")
            if quiz_type != "open-ended":
                options = st.session_state.quiz_data.alternatives[idx]
                user_answer_key = st.radio("Select an answer:", options, key=idx)
                user_answers[idx] = user_answer_key
            else:
                user_answers[idx] = st.text_area("Your answer:", key=idx)

        if st.button("Score Quiz"):
            score = 0
            correct_answers = []
            for idx, question in enumerate(st.session_state.quiz_data.questions):
                correct_answer = st.session_state.quiz_data.answers[idx]
                if quiz_type != "open-ended":
                    if user_answers[idx] == correct_answer:
                        score += 1
                correct_answers.append(f"{idx + 1}. {correct_answer}")
            st.write("Quiz results:")
            st.write(f"Your score: {score}/{len(st.session_state.quiz_data.questions)}")
            for correct_answer in correct_answers:
                st.write(correct_answer)


if __name__ == "__main__":
    main()

