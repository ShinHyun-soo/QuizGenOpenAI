import streamlit as st

# Page title and instructions
st.title("QuizGen :books:")
st.write("좌측에서 첨부할 문제 내용을 선택하여 주시기 바랍니다.")

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
    background-color: white;
    text-align: center;
    padding: 10px 0;
    font-size: 14px;
    color: #333;
}
</style>
"""

# Footer HTML
footer_html = """
<div class="footer">
  <p>Copyright 2024. QuizGen Co. all rights reserved.</p>
</div>
"""

# Inject CSS with markdown
st.markdown(footer_css, unsafe_allow_html=True)

# Inject footer HTML with markdown
st.markdown(footer_html, unsafe_allow_html=True)
