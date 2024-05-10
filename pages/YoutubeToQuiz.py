from langchain.document_loaders import YoutubeLoader
from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import streamlit as st
import openai

# Display the Page Title
st.title(' Youtube QnA')

load_dotenv()

llm = OpenAI()  # Temp controls the randomness of the text

prompt = st.text_input("Paste the URL")

if prompt:
    loader = YoutubeLoader.from_youtube_url(prompt, add_video_info=False)
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)

    split_docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    doc_search = Chroma.from_documents(docs,embeddings)
    #Stuff: 단순히 모든 문서를 단일 프롬프트로 “넣는” 방식입니다. 이는 가장 간단한 접근 방식입니다.
    # Map - reduce: 각 문서를 “map” 단계에서 개별적으로 요약한 다음, “reduce” 단계에서 요약본들을 최종 요약본으로 합치는 방식입니다.
    # Refine: 입력 문서를 순회하며 반복적으로 답변을 업데이트하여 응답을 구성합니다. 각 문서에 대해, 모든 비문서 입력, 현재 문서, 그리고 최신 중간 답변을 LLM chain에 전달하여 새로운 답변을 얻습니다.
    chain = RetrievalQA.from_chain_type(llm=llm,chain_type='stuff',retriever=doc_search.as_retriever())

    query = "Create quiz based on"


    if st.button("문제 생성"):
        answer = chain.run(query)

        prompt_with_translation = f"Translate the following into Korean:\n{answer}\n"

        translation_response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",  # Choose an engine that supports translation
            prompt=prompt_with_translation,
            temperature=0,
            max_tokens=600,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            #stop=["\n"]
        )
        translated_answer = translation_response.choices[0].text.strip()

        st.session_state.script = translated_answer
        st.text_area("Generated quiz", st.session_state.script, height=500)


