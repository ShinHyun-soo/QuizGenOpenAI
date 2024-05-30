import streamlit as st

from streamlit_option_menu import option_menu
import os
# from dotenv import load_dotenv
# load_dotenv()

import auth, kwd, pdf, txt, url, ytb

# st.set_page_config(
#     page_title="Pondering",
# )

st.markdown(
    """
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src=f"https://www.googletagmanager.com/gtag/js?id={os.getenv('analytics_tag')}"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', os.getenv('analytics_tag'));
        </script>
    """, unsafe_allow_html=True)
print(os.getenv('analytics_tag'))


class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:
            app = option_menu(
                menu_title='문제 출제 소스 ',
                options=['토픽/주제', '텍스트 입력',  'PDF 파일', 'URL 주소', 'YouTube 링크', '로그인'],
                icons=['chat-text-fill', 'chat-text-fill', 'file', 'link', 'link', 'person-circle'],
                menu_icon=None,
                default_index=0,
                styles={
                    "container": {"padding": "5!important", "background-color": 'lightgray'},
                    "icon": {"color": "black", "font-size": "18px"},
                    "nav-link": {"color": "black", "font-size": "20px", "text-align": "left", "margin": "0px",
                                 "--hover-color": "lightgreen"},
                    "nav-link-selected": {"background-color": "#02ab21"}, }

            )

        if app == "토픽/주제":
            kwd.main()
        if app == "텍스트 입력":
            txt.main()
        if app == "PDF 파일":
            pdf.main()
        if app == 'URL 주소':
            url.main()
        if app == 'YouTube 링크':
            ytb.main()
        if app == '로그인':
            auth.main()

    run()
