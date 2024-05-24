import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field

class QuizMultipleChoice(BaseModel):
    quiz_text: str = Field(description="The quiz test")
    questions: List[str] = Field(description="The quiz quesitons")
    alternatives: List[List[str]] = Field(description="The quiz alternatives")
    answers: List[str] = Field(description="The quiz answers")

class QuizTrueFalse(BaseModel):
    quiz_text: str = Field(description="The quiz test")
    questions: List[str] = Field(description="The quiz quesitons")
    alternatives: List[List[str]] = Field(description="The quiz alternatives")
    answers: List[str] = Field(description="The quiz answers")

class QuizOpenEnded(BaseModel):
    questions: List[str] = Field(description="The quiz quesitons")
    answers: List[str] = Field(description="The quiz answers")

def create_quiz_chain(prompt_template, llm, pydantic_object_schema):
    """Creates the chain for the quiz app."""
    return prompt_template | llm.with_structured_output(pydantic_object_schema)

def create_multiple_choice_template(language, user_input=None):
    """Create the prompt template for the quiz app, including conditional translation."""
    template = """ 
    You are an expert quiz maker for technical fields. Let's think step by step and
    create a {difficulty} quiz with {num_questions} multiple-choice questions about the following concept/content: {quiz_context}.

    The format of the quiz should be as follows:
    
    - Multiple-choice: 
    - Questions:
        <Question1>: 
            - Alternatives1: <option 1>, <option 2>, <option 3>, <option 4>
        <Question2>: 
            - Alternatives2: <option 1>, <option 2>, <option 3>, <option 4>
        ....
        <QuestionN>: 
            - AlternativesN: <option 1>, <option 2>, <option 3>, <option 4>
    - Answers:
        <Answer1>: <option 1 | option 2 | option 3 | option 4>
        <Answer2>: <option 1 | option 2 | option 3 | option 4>
        ....
        <AnswerN>: <option 1 | option 2 | option 3 | option 4>
    """

    # Conditionally add translation instruction based on the selected language
    if language != "English":
        template += f"\n\nPlease ensure that the quiz is accurately translated into {language}, maintaining the technical accuracy and clarity of the questions and options."

    if user_input != None:
        template += f"\n\n {user_input}"


    prompt = ChatPromptTemplate.from_template(template)
    return prompt



def create_true_false_template(language, user_input=None):
    """Create the prompt template for the quiz app."""
    
    template = """
    You are an expert quiz maker for technical fields. Let's think step by step and
    create a {difficulty} quiz with {num_questions} questions about the following concept/content: {quiz_context}.

    The format of the quiz could be one of the following:
    - True-false:

    - Questions:
        <Question1>: 
            - Alternatives1: <True>, <False>
        <Question2>: 
            - Alternatives2: <True>, <False>
        .....
        <QuestionN>: 
            - AlternativesN: <True>, <False>
    - Answers:
        <Answer1>: <True|False>
        <Answer2>: <True|False>
        .....
        <AnswerN>: <True|False>
    """
    # Conditionally add translation instruction based on the selected language
    if language != "English":
        template += f"\n\nPlease ensure that the quiz is accurately translated into {language}, maintaining the technical accuracy and clarity of the questions and options."

    if user_input != None:
        template += f"\n\n {user_input}"

    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def create_open_ended_template(language, user_input=None):
    template = """
    You are an expert quiz maker for technical fields. Let's think step by step and
    create a {difficulty} quiz with {num_questions} questions about the following concept/content: {quiz_context}.

    The format of the quiz could be one of the following:
    - Open-ended:
    - Questions:
        <Question1>: 
        <Question2>:
        .....
        <QuestionN>:
    - Answers:    
        <Answer1>:
        <Answer2>:
        .....
        <AnswerN>:
       
    """
    # Conditionally add translation instruction based on the selected language
    if language != "English":
        template += f"\n\nPlease ensure that the quiz is accurately translated into {language}, maintaining the technical accuracy and clarity of the questions and options."

    if user_input != None:
        template += f"\n\n {user_input}"

    prompt = ChatPromptTemplate.from_template(template)
    return prompt




def main():
    st.header("QuizGen:books:")
    st.caption("키워드 입력 후 원하시는 문제 유형을 선택하여 주십시오.")
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    on = st.toggle("GPT-4o")
    if on:
        llm = ChatOpenAI(model="gpt-4o")
    context = st.text_area("키워드를 입력해 주십시오.")
    col1, col2, col3 = st.columns(3)

    # 첫 번째 컬럼에 난이도 선택 라디오 버튼을 배치합니다.
    with col3:
        difficulty = st.radio("난이도", ["easy", "normal", "hard"])

    # 두 번째 컬럼에 언어 선택 라디오 버튼을 배치합니다.
    with col1:
        language = st.radio("언어 선택", ["Korean", "English"])  # 언어 선택

    # 세 번째 컬럼에 종류 선택 라디오 버튼을 배치합니다.
    with col2:
        quiz_type = st.radio("종류 선택", ["객관식", "참/거짓", "주관식"])
    num_questions = st.number_input("갯수 선택", min_value=1, max_value=10, value=3)
    user_input = st.text_area("기타 요구 사항을 입력해 주십시오.")

    # 퀴즈 유형 변경 시 상태 초기화
    if 'quiz_type' not in st.session_state or st.session_state.quiz_type != quiz_type:
        st.session_state.quiz_type = quiz_type
        st.session_state.quiz_data = None
        st.session_state.user_answers = None


    if st.button("퀴즈 생성"):
        if quiz_type == "객관식":
            prompt_template = create_multiple_choice_template(language, user_input)
            pydantic_object_schema = QuizMultipleChoice
        elif quiz_type == "참/거짓":
            prompt_template = create_true_false_template(language, user_input)
            pydantic_object_schema = QuizTrueFalse
        else:
            prompt_template = create_open_ended_template(language, user_input)
            pydantic_object_schema = QuizOpenEnded

        st.write("퀴즈 생성 중, 올바르게 생성되지 않으면 퀴즈 생성 버튼을 다시 눌러 주시기 바랍니다.")
        chain = create_quiz_chain(prompt_template, llm, pydantic_object_schema)
        st.session_state.quiz_data = chain.invoke({"num_questions": num_questions, "quiz_context": context, "difficulty": difficulty})
        st.session_state.user_answers = [None] * len(st.session_state.quiz_data.questions) if st.session_state.quiz_data else []


    if 'quiz_data' in st.session_state and st.session_state.quiz_data:
        user_answers = {}
        for idx, question in enumerate(st.session_state.quiz_data.questions):
            st.write(f"**{idx + 1}. {question}**")
            if quiz_type != "주관식":
                options = st.session_state.quiz_data.alternatives[idx]
                user_answer_key = st.radio("답 : ", options, key=idx)
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

    footer_css = """
    <style>
    # MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        color: #333;
    }
    </style>
    """

    # Footer HTML background-color: white;
    footer_html = """
    <div class="footer">
      <p>ⓒ 2024. QuizGen. all rights reserved.</p>
    </div>
    """

    # Inject CSS with markdown
    st.markdown(footer_css, unsafe_allow_html=True)

    # Inject footer HTML with markdown
    st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

