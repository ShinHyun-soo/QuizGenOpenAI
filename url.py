import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain
from common import (
    load_llm,
    initialize_session_state,
    export_quiz_data,
    create_gift_format,
    create_xhtml_format,
    create_xml_format
)
import sys
sys.path.append('/')  # 상위 폴더를 시스템 경로에 추가
from promptTemplates import QuizMultipleChoice, QuizTrueFalse, QuizOpenEnded, QuizShortAnswer, create_quiz_chain,create_multiple_choice_template, create_true_false_template, create_open_ended_template, create_short_answer_template
from htmlTemplates import css, footer_css, footer_html


##임시 업로드용 파일 TextToQuiz 구현 후 대체 해야함.

def get_text_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    return document

def process_text_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(text)
    return document_chunks

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def select_chunk_set(vectorstore, text_chunks, num_vectors=5):
    # 구현의 편의를 위해 앞쪽 5개의 텍스트 가져옴
    chunks = text_chunks[:5]
    # 선택된 텍스트 청크들을 하나의 문자열로 결합
    context = ' '.join(chunk.page_content for chunk in chunks)
    return context

def main():
    # app config
    #st.set_page_config(page_title="사이트 기반 문제 생성", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)
    st.header("QuizGen :books:")
    st.caption("url 입력 후 원하시는 문제를 선택하여 주십시오. ")
    initialize_session_state(st.session_state, 4)
    website_url = st.text_input("Url 입력란")

    if st.button("입력"):
        with st.spinner("입력 중"):
            raw_text = get_text_from_url(website_url)

            text_chunks = process_text_to_chunks(raw_text)

            vectorstore = create_vector_store(text_chunks)

            st.session_state.context4 = select_chunk_set(vectorstore, text_chunks)

            st.success('저장 완료!', icon="✅")

            expander = st.expander("내용 확인")
            expander.write(raw_text)

    col1, col2, col3, col4 = st.columns(4)

    # 첫 번째 컬럼에 난이도 선택 라디오 버튼을 배치합니다.
    with col3:
        difficulty = st.radio("난이도", ["easy", "normal", "hard"], index=1)

    # 두 번째 컬럼에 언어 선택 라디오 버튼을 배치합니다.
    with col1:
        language = st.radio("언어 선택", ["Korean", "English"])  # 언어 선택

    # 세 번째 컬럼에 종류 선택 라디오 버튼을 배치합니다.
    with col2:
        quiz_type = st.radio("종류 선택", ["객관식", "참/거짓", "주관식", "단답형"])

    with col4:
        llm_type = st.radio("LLM", ["Llama-3", "GPT-3.5-Turbo", "GPT-4", "GPT-4o"],index=3)

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

    if "context" not in st.session_state:
        st.session_state.context = None  # context 초기화

    # 퀴즈 유형 변경 시 상태 초기화
    if 'quiz_type' not in st.session_state or st.session_state.quiz_type != quiz_type:
        st.session_state.quiz_type = quiz_type
        st.session_state.quiz_data = None
        st.session_state.user_answers = None

    if st.button("문제 생성"):
        if st.session_state.context4:
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

            st.write("에러가 발생할 경우, 다시 생성버튼을 눌러주시면 됩니다.")
            chain = create_quiz_chain(prompt_template, llm, pydantic_object_schema)
            st.session_state.quiz_data4 = chain.invoke(
                {"num_questions": num_questions, "quiz_context": st.session_state.context4, "difficulty": difficulty, "user_input":user_input})
            st.session_state.user_answers4 = [None] * len(
                st.session_state.quiz_data4.questions) if st.session_state.quiz_data4 else []
            st.session_state.quiz_generated4 = True
            st.session_state.show_results4 = False
        else:
            st.write("url이 입력 되지 않았습니다.")

    if st.session_state.quiz_generated4 and st.session_state.quiz_data4:
        for idx, question in enumerate(st.session_state.quiz_data4.questions):
            st.write(f"**{idx + 1}. {question}**")
            if quiz_type == "객관식" or quiz_type == "참/거짓":
                options = st.session_state.quiz_data4.alternatives[idx]
                st.session_state.user_answers4[idx] = st.radio("답:", options, key=f"user_answer4_{idx}", index=None)
                if st.session_state.show_results4:
                    correct_answer = st.session_state.quiz_data4.answers[idx]
                    if st.session_state.user_answers4[idx] == correct_answer:
                        st.success(f"정답: {correct_answer}")
                    else:
                        st.error(f"정답: {correct_answer}")
            elif quiz_type == "단답형":
                st.session_state.user_answers4[idx] = st.text_input("답:", key=f"user_answer4_{idx}")
                if st.session_state.show_results4:
                    correct_answer = st.session_state.quiz_data4.answers[idx]
                    if st.session_state.user_answers4[idx].strip() == correct_answer.strip():
                        st.success(f"정답: {correct_answer}")
                    else:
                        st.error(f"정답: {correct_answer}")
            else:
                st.session_state.user_answers4[idx] = st.text_area("답:", key=f"user_answer4_{idx}")
                if st.session_state.show_results4:
                    correct_answer = st.session_state.quiz_data4.answers[idx]
                    if st.session_state.user_answers4[idx].strip() == correct_answer.strip():
                        st.success(f"정답: {correct_answer}")
                    else:
                        st.error(f"정답: {correct_answer}")

        if st.button("채점"):
            score = 0
            correct_answers = []
            for idx, question in enumerate(st.session_state.quiz_data4.questions):
                correct_answer = st.session_state.quiz_data4.answers[idx]
                if quiz_type != "주관식":
                    if st.session_state.user_answers4[idx] == correct_answer:
                        score += 1
                correct_answers.append(f"{idx + 1}. {correct_answer}")

            st.session_state.show_results4 = True
            st.session_state.score4 = score
            st.session_state.correct_answers4 = correct_answers
            st.rerun()

    if st.session_state.show_results4:
        st.subheader("채점 결과")
        st.write(f"점수: {st.session_state.score4}/{len(st.session_state.quiz_data4.questions)}")
        expander = st.expander("정답 보기")
        for correct_answer in st.session_state.correct_answers4:
            expander.write(correct_answer)

    if st.session_state.quiz_data4:
        export_format = st.selectbox("내보낼 형식을 선택하세요", ["JSON", "CSV", "TXT", "GIFT", "XHTML", "XML"])
        data, mime, filename = export_quiz_data(st.session_state.quiz_data4, st.session_state.context4, export_format)
        if data and mime and filename:
            st.download_button(label=f"퀴즈 다운로드 ({export_format})", data=data, file_name=filename, mime=mime)

    st.markdown(footer_css, unsafe_allow_html=True)

    # Inject footer HTML with markdown
    st.markdown(footer_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
