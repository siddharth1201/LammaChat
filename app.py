import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

## Langsmith Tracking
api_key = os.getenv("LANGCHAIN_API_KEY")
project_name = os.getenv("LANGCHAIN_PROJECT")

if api_key is None:
    st.error("LANGCHAIN_API_KEY environment variable is not set. Please set it in your .env file.")
else:
    os.environ["LANGCHAIN_API_KEY"] = api_key

if project_name:
    os.environ["LANGCHAIN_PROJECT"] = project_name

os.environ["LANGCHAIN_TRACING_V2"] = "true"

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question: {question}")
    ]
)

## Streamlit framework
st.title("Langchain Demo With llama3.1 Model")
input_text = st.text_input("What question do you have in mind?")

## Ollama Llama2 model
llm = Ollama(model="llama3.1")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

## Execute the chain if the user provides input
if input_text:
    try:
        response = chain.invoke({"question": input_text})
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
