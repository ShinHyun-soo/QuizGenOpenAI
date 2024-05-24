import streamlit as st

# Page title and instructions

st.title("QuizGen :books:")

st.write("좌측에서 문제 생성에 참고할 파일 유형을 선택하여 주십시오.")

st.caption("Sponsored by")

st.image('hsu.png', width=200)

from streamlit_google_auth import Authenticate

#google_credentials = st.secrets["GOOGLE_CREDENTIALS"]


authenticator = Authenticate(
    secret_credentials_path = 'google_credentials.json',
    cookie_name='my_cookie_name',
    cookie_key='this_is_secret',
    redirect_uri = 'https://quiz-bot-4.streamlit.app/',
)

# Catch the login event
authenticator.check_authentification()

# Create the login button
authenticator.login()

if st.session_state['connected']:
    st.image(st.session_state['user_info'].get('picture'))
    st.write('Hello, '+ st.session_state['user_info'].get('name'))
    st.write('Your email is '+ st.session_state['user_info'].get('email'))
    if st.button('Log out'):
        authenticator.logout()

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



