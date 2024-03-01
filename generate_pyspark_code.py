#generate_pyspark_code.py
import pandas as pd
import openai
import os
import tiktoken
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from dotenv import load_dotenv
from openai import OpenAI
from transformers import GPT2Tokenizer


def call_llm(prompt):
    client = OpenAI()
    messages = [
        {"role": "system", "content": "You are a pyspark code generation assistant."},
        {"role": "user", "content": prompt}  # Your existing prompt becomes the user's message content
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125", #gpt-4-turbo-preview #gpt-3.5-turbo-0613
        messages=messages,
        temperature=0.5,
        max_tokens=4096,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    if response.choices:
        last_message = response.choices[0].message  # This might need adjustment based on the actual structure
        generated_code = last_message.content if hasattr(last_message, 'content') else ""
    else:
        generated_code = ""
    
    return generated_code.strip()


def generate_pyspark_code_from_stm(df: pd.DataFrame):
    """
    Generates PySpark code for data transformations specified in a DataFrame.
    """
    # Construct the prompt for the LLM
    prompt = "STM contains 3 fields as Source, Transformation and Target as below. Could you write the pyspark code to apply transformations and write it into the target table and the target database is hive and table name is hive table:\n"
    
    df.columns = df.columns.str.strip()
    print(df.columns)  # This will print all column names

    generated_code = ""
    complete_prompt = ""
    # Outer loop to iterate through the DataFrame in steps of 5 rows
    for start_index in range(0, len(df), 3):
        # Reset prompt for each slice
        prompt = ""
        # Inner loop to process 5 rows at a time including the current row from the outer loop
        end_index = min(len(df), start_index + 3)  # Ensure the end index does not exceed the length of the DataFrame
        # Slice the DataFrame for the next 5 rows including the current one
        df_slice = df.iloc[start_index:end_index]
        # Process the sliced DataFrame (df_slice) here
        # For example, print the sliced DataFrame
        print(df_slice)
        for _, row in df_slice.iterrows():
            source, transformation, target = row['Source'], row['Transformation'], row['Target'].strip()
            prompt += f"- Apply transformation '{transformation}' on column '{source}' and store in '{target}'\n"
        generated_code += call_llm(prompt)
        complete_prompt += prompt
        if end_index >= len(df):
            break

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
    print(encoding)
    input_tokens = num_tokens_from_string(complete_prompt, "cl100k_base")
    # print(prompt)
    # Call the LLM with the constructed prompt
    #generated_code = call_llm(prompt)
    output_tokens = num_tokens_from_string(generated_code, "cl100k_base")
    total_tokens = input_tokens + output_tokens
    print(f"input tokens: {input_tokens}")
    print(f"output tokens: {output_tokens}")
    print(f"total tokens: {total_tokens}")
    #print(tiktoken.encoding_for_model("gpt-3.5-turbo-0125"))
    #print(f" prompt encoding: {encoding.decode(encoding.encode(prompt))}")
    #print(f" code encoding: {encoding.decode(encoding.encode(generated_code))}")
    return generated_code

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens




    

    
  
    







