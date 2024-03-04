#generate_pyspark_code.py
import pandas as pd
import openai
import os
import tiktoken
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr
from dotenv import load_dotenv
from openai import OpenAI


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

def remove_specific_lines(code_str):
    
    lines = code_str.split('\n') # Split the string into lines
    filtered_lines = [line for line in lines if not line.startswith('```') and not line.startswith('from') and not line.startswith('Make sure')] # Filter out lines that start with ```python or "from" or "Make sure"
    filtered_code_str = '\n'.join(filtered_lines)  # Join the filtered lines back into a string
    return filtered_code_str


def adjust_final_full_code(filtered_code_str):
    adjusted_code_str = ""
    adjusted_code_str += f"""```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Create Spark session
spark = SparkSession.builder.appName("STM Transformation").enableHiveSupport().getOrCreate()

# Load data from source table
source_df = spark.table("source_table")
{filtered_code_str}
# Write transformed data to target table in Hive
transformed_df.write.mode("overwrite").saveAsTable("hive_table")

# Stop Spark session
spark.stop()
``` 

Please replace 'source_table' with the actual name of your source table in the Hive database."""
    return adjusted_code_str

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

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
    final_code = ""
    filtered_code = ""
        
    for _, row in df.iterrows(): # Construct the complete prompt
        source, transformation, target = row['Source'], row['Transformation'], row['Target'].strip()
        prompt += f"- Apply transformation '{transformation}' on column '{source}' and store in '{target}'\n"
        
    input_tokens = num_tokens_from_string(prompt, "cl100k_base")
    if input_tokens < 2500:
        generated_code = call_llm(prompt)
        final_code = generated_code
            
        filename = 'simple_final_code.txt'
        with open(filename, 'w') as file:
            file.write(final_code)
    else:
        for start_index in range(0, len(df), 3):
            prompt = "" # Reset prompt for each slice
            end_index = min(len(df), start_index + 3)  
            df_slice = df.iloc[start_index:end_index]  
            print(df_slice) # For example, print the sliced DataFrame
            
            for _, row in df_slice.iterrows():
                source, transformation, target = row['Source'], row['Transformation'], row['Target'].strip()
                prompt += f"- Apply transformation '{transformation}' on column '{source}' and store in '{target}'\n"
            generated_code += call_llm(prompt)
            complete_prompt += prompt
            if end_index >= len(df):
                break
        filtered_code = remove_specific_lines(generated_code)
        final_code = adjust_final_full_code(filtered_code)

        filename = 'generated_code.txt'
        with open(filename, 'w') as file:
            file.write(generated_code)

        filename = 'final_code.txt'
        with open(filename, 'w') as file:
            file.write(final_code)
            
    # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
    # print(encoding)
    #input_tokens = num_tokens_from_string(complete_prompt, "cl100k_base")
    output_tokens = num_tokens_from_string(generated_code, "cl100k_base")
    total_tokens = input_tokens + output_tokens
    print(f"input tokens: {input_tokens}")
    print(f"output tokens: {output_tokens}")
    print(f"total tokens: {total_tokens}")
    return final_code






    

    
  
    







