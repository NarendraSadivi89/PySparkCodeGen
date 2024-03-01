# importing dependencies
import streamlit as st
import os
import pandas as pd
import openai  
from openai import OpenAI
from dotenv import load_dotenv
from generate_pyspark_code import generate_pyspark_code_from_stm
from htmlTemplates import css, bot_template, user_template


class PySparkChatPage:
   def __init__(
            self,
            page_title,
            page_icon,
            header
    ):
        load_dotenv()
        st.title("PySpark Code Generator from STM")
        st.write("Upload an STM (Excel) and get PySpark code for transformations.")

        uploaded_file = st.file_uploader("Choose an STM file", type=["csv"])
        pyspark_code = ""
        if uploaded_file is not None:
        # Process the STM file
            try:
                stm_df = pd.read_csv(uploaded_file)
                st.write("STM Preview:")
                st.dataframe(stm_df.head())  # Displaying a preview of the STM file
                st.write("Please click on 'Generate PySpark Code' button first and then click on 'Dowload Code' button ")
                # Button to generate PySpark code
                if st.button("Generate PySpark Code"):
                    pyspark_code = generate_pyspark_code_from_stm(stm_df)  # Generate code using the backend function
                    st.text_area("Generated PySpark Code", pyspark_code, height=800)
                    st.download_button("Download Code", pyspark_code, file_name="pyspark_code.py")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")

        
        
if __name__ == '__main__':
    PySparkChatPage(
        page_title="Chat",
        page_icon="🤖",
        header="Ask your question"
    )
