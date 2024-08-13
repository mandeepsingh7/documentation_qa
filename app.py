from query import get_final_response
import streamlit as st


def main():
    st.set_page_config(
        page_title="Cuda Documentation QA",
        page_icon="📁",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items={
            'Get Help': 'https://docs.nvidia.com/cuda/',
            'Report a bug': "https://github.com/mandeepsingh7",
            'About': "# Created by Mandeep Singh"
        }
    )
    st.header("NVIDIA Cuda Documentation QA")

    user_question = st.text_input("Ask a Question")

    if user_question:
        response = get_final_response(user_question)
        st.write('Response:', response['response_content'])
        st.write('Source titles:', response['source_titles'])
        st.write('Source URLs:', response['source_urls'])


if __name__ == "__main__":
    main()
