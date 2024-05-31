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
from promptTemplates import  QuizShortAnswer, QuizMultipleChoice, QuizTrueFalse, QuizOpenEnded, create_quiz_chain, create_multiple_choice_template, create_true_false_template, create_open_ended_template, create_short_answer_template
from htmlTemplates import css
from htmlTemplates import css, footer_css, footer_html
from common import (
    load_llm,
    initialize_session_state,
    export_quiz_data,
    create_gift_format,
    create_xhtml_format,
    create_xml_format
)



def main():
    st.header("QuizGen:books:")
    st.caption("키워드 입력 후 원하시는 문제 유형을 선택하여 주십시오.")
    context = st.text_input("키워드를 입력해 주십시오. 예) 랭체인")
    col1, col2, col3, col4 = st.columns(4)

    # 첫 번째 컬럼에 난이도 선택 라디오 버튼을 배치합니다.
    with col3:
        difficulty = st.radio("난이도", ["easy", "normal", "hard"], index=1)

    # 두 번째 컬럼에 언어 선택 라디오 버튼을 배치합니다.
    with col1:
        language = st.radio("언어 선택", ["Korean", "English"], index=0)  # 언어 선택

    # 세 번째 컬럼에 종류 선택 라디오 버튼을 배치합니다.
    with col2:
        quiz_type = st.radio("종류 선택", ["객관식", "참/거짓", "주관식", "단답형"], index=0)

    with col4:
        llm_type = st.radio("LLM", ["Llama-3(준비 중)", "GPT-3.5-Turbo", "GPT-4", "GPT-4o"], index=3)
    num_questions = st.number_input("갯수 선택", min_value=1, max_value=10, value=3)
    user_input = st.text_area("기타 요구 사항을 입력해 주십시오.")

    if llm_type == "Llama-3":
        llm = ChatOpenAI(model="gpt-4o")

    if llm_type == "GPT-3.5-Turbo":
        llm = ChatOpenAI(model="gpt-3.5-turbo")

    if llm_type == "GPT-4":
        llm = ChatOpenAI(model="gpt-4")

    if llm_type == "GPT-4o":
        llm = ChatOpenAI(model="gpt-4o")

    initialize_session_state(st.session_state, 1)
    # 퀴즈 유형 변경 시 상태 초기화
    if st.button("퀴즈 생성"):
        if context.strip() == "":
            st.warning("키워드를 입력해 주십시오.")
        else:
            if quiz_type == "객관식":
                prompt_template = create_multiple_choice_template(language)
                pydantic_object_schema = QuizMultipleChoice
            elif quiz_type == "참/거짓":
                prompt_template = create_true_false_template(language)
                pydantic_object_schema = QuizTrueFalse
            elif quiz_type == "단답형":
                prompt_template = create_short_answer_template(language)
                pydantic_object_schema = QuizShortAnswer
            else:
                prompt_template = create_open_ended_template(language)
                pydantic_object_schema = QuizOpenEnded
            st.write("(생성 중), 퀴즈가 올바르게 생성되지 않으면, 기타 요구 사항란을 이용하여 퀴즈를 생성해 보시기 바랍니다.")
            chain = create_quiz_chain(prompt_template, llm, pydantic_object_schema)
            st.session_state.quiz_data1 = chain.invoke(
                {"num_questions": num_questions, "quiz_context": context, "difficulty": difficulty, "user_input": user_input})
            st.session_state.user_answers1 = [None] * len(
                st.session_state.quiz_data1.questions) if st.session_state.quiz_data1 else []
            st.session_state.quiz_generated1 = True
            st.session_state.show_results1 = False

    if st.session_state.quiz_generated1 and st.session_state.quiz_data1:
        for idx, question in enumerate(st.session_state.quiz_data1.questions):
            st.write(f"**{idx + 1}. {question}**")
            if quiz_type == "객관식" or quiz_type == "참/거짓":
                options = st.session_state.quiz_data1.alternatives[idx]
                st.session_state.user_answers1[idx] = st.radio("답 : ", options, key=f"user_answer1_{idx}", index=None)
                if st.session_state.show_results1:
                    correct_answer = st.session_state.quiz_data1.answers[idx]
                    if st.session_state.user_answers1[idx] == correct_answer:
                        st.success(f"정답: {correct_answer}")
                    else:
                        st.error(f"정답: {correct_answer}")
            elif quiz_type == "단답형":
                st.session_state.user_answers1[idx] = st.text_input("답 : ", key=f"user_answer1_{idx}")
                if st.session_state.show_results1:
                    correct_answer = st.session_state.quiz_data1.answers[idx]
                    if st.session_state.user_answers1[idx].strip() == correct_answer.strip():
                        st.success(f"정답: {correct_answer}")
                    else:
                        st.error(f"정답: {correct_answer}")
            else:
                st.session_state.user_answers1[idx] = st.text_area("답 : ", key=f"user_answer1_{idx}")
                if st.session_state.show_results1:
                    correct_answer = st.session_state.quiz_data1.answers[idx]
                    if st.session_state.user_answers1[idx].strip() == correct_answer.strip():
                        st.success(f"정답: {correct_answer}")
                    else:
                        st.error(f"정답: {correct_answer}")

        if st.button("채점"):
            score = 0
            correct_answers = []
            for idx, question in enumerate(st.session_state.quiz_data1.questions):
                correct_answer = st.session_state.quiz_data1.answers[idx]
                if quiz_type != "주관식":
                    if st.session_state.user_answers1[idx] == correct_answer:
                        score += 1
                correct_answers.append(f"{idx + 1}. {correct_answer}")

            st.session_state.show_results1 = True
            st.session_state.score1 = score
            st.session_state.correct_answers1 = correct_answers
            st.rerun()

    if st.session_state.show_results1:
        st.subheader("채점 결과")
        st.write(f"점수: {st.session_state.score1}/{len(st.session_state.quiz_data1.questions)}")
        expander = st.expander("정답 보기")
        for correct_answer in st.session_state.correct_answers1:
            expander.write(correct_answer)

    if st.session_state.quiz_data1:
        export_format = st.selectbox("내보낼 형식을 선택하세요", ["JSON", "CSV", "TXT", "GIFT", "XHTML", "XML"])
        data, mime, filename = export_quiz_data(st.session_state.quiz_data1, context, export_format)
        if data and mime and filename:
            st.download_button(label=f"퀴즈 다운로드 ({export_format})", data=data, file_name=filename, mime=mime)

    st.markdown(footer_css, unsafe_allow_html=True)
    st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

