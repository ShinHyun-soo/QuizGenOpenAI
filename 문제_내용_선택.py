import streamlit as st

# Page title and instructions

st.title("QuizGen :books:")

st.write("좌측에서 문제 생성에 참고할 파일 유형을 선택하여 주십시오.")

st.caption("Sponsored by")

st.image('hsu.png', width=200)

# Custom CSS for the footer
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



