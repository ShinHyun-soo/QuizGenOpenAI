import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field

import sys
sys.path.append('/')  # 상위 폴더를 시스템 경로에 추가
from promptTemplates import QuizMultipleChoice, QuizTrueFalse, QuizOpenEnded, create_quiz_chain, create_multiple_choice_template, create_true_false_template, create_open_ended_template
from htmlTemplates import css
from htmlTemplates import css, footer_css, footer_html



def main():
    st.header("QuizGen:books:")
    st.caption("키워드 입력 후 원하시는 문제 유형을 선택하여 주십시오.")
    context = st.text_input("키워드를 입력해 주십시오. 예) 랭체인")
    col1, col2, col3, col4 = st.columns(4)

    # 첫 번째 컬럼에 난이도 선택 라디오 버튼을 배치합니다.
    with col3:
        difficulty = st.radio("난이도", ["easy", "normal", "hard"])

    # 두 번째 컬럼에 언어 선택 라디오 버튼을 배치합니다.
    with col1:
        language = st.radio("언어 선택", ["Korean", "English"])  # 언어 선택

    # 세 번째 컬럼에 종류 선택 라디오 버튼을 배치합니다.
    with col2:
        quiz_type = st.radio("종류 선택", ["객관식", "참/거짓", "주관식"])

    with col4:
        llm_type = st.radio("LLM", ["Llama-3", "GPT-3.5-Turbo", "GPT-4", "GPT-4o"])
    num_questions = st.number_input("갯수 선택", min_value=1, max_value=10, value=3)
    user_input = st.text_area("기타 요구 사항을 입력해 주십시오.")

    if llm_type == "Llama-3":
        llm = ChatOpenAI(model="gpt-4o")

    if llm_type == "GPT-3.5-Turbo":
        llm = ChatOpenAI(model="gpt-4o")

    if llm_type == "GPT-4":
        llm = ChatOpenAI(model="gpt-4o")

    if llm_type == "GPT-4o":
        llm = ChatOpenAI(model="gpt-4o")

    # 퀴즈 유형 변경 시 상태 초기화
    if 'quiz_type' not in st.session_state or st.session_state.quiz_type != quiz_type:
        st.session_state.quiz_type = quiz_type
        st.session_state.quiz_data = None
        st.session_state.user_answers = None


    if st.button("퀴즈 생성"):
        if quiz_type == "객관식":
            prompt_template = create_multiple_choice_template(language)
            pydantic_object_schema = QuizMultipleChoice
        elif quiz_type == "참/거짓":
            prompt_template = create_true_false_template(language)
            pydantic_object_schema = QuizTrueFalse
        else:
            prompt_template = create_open_ended_template(language)
            pydantic_object_schema = QuizOpenEnded

        st.write("퀴즈 생성 중, 올바르게 생성되지 않으면 퀴즈 생성 버튼을 다시 눌러 주시기 바랍니다.")
        chain = create_quiz_chain(prompt_template, llm, pydantic_object_schema)
        st.session_state.quiz_data = chain.invoke({"num_questions": num_questions, "quiz_context": context, "difficulty": difficulty, "user_input": user_input})
        st.session_state.user_answers = [None] * len(st.session_state.quiz_data.questions) if st.session_state.quiz_data else []


    if 'quiz_data' in st.session_state and st.session_state.quiz_data:
        user_answers = {}
        for idx, question in enumerate(st.session_state.quiz_data.questions):
            st.write(f"**{idx + 1}. {question}**")
            if quiz_type != "주관식":
                options = st.session_state.quiz_data.alternatives[idx]
                user_answer_key = st.radio("답 : ", options, key=idx, index=None)
                user_answers[idx] = user_answer_key
            else:
                user_answers[idx] = st.text_area("답 : ", key=idx)

        if st.button("채점"):
            score = 0
            correct_answers = []
            for idx, question in enumerate(st.session_state.quiz_data.questions):
                correct_answer = st.session_state.quiz_data.answers[idx]
                if quiz_type != "주관식":
                    if user_answers[idx] == correct_answer:
                        score += 1
                correct_answers.append(f"{idx + 1}. {correct_answer}")
            st.subheader("채점 결과")
            st.write(f"점수: {score}/{len(st.session_state.quiz_data.questions)}")
            expander = st.expander("정답 보기")
            for correct_answer in correct_answers:
                expander.write(correct_answer)



    # Inject CSS with markdown
    st.markdown(footer_css, unsafe_allow_html=True)

    # Inject footer HTML with markdown
    st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

