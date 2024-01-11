import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
import os
from llama_index import SimpleDirectoryReader

file_path = './Corpus1' 
st.set_page_config(page_title="Essay Writer", page_icon="✍️", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with JC's MBA Essays")
st.subheader("JC has uploaded his MBA admissions essays. Ask questions about him or his writing!", divider=True)
st.info("Some example prompts:  \nWhat is JC's leadership philosophy?  \nIn JC's voice, write a 10-sentence essay on a time he demonstrated leadership.")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about JC and his essays!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir=file_path, recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4", system_prompt="You are a helpful assistant answering questions about the files uploaded by JC, his MBA admissions essays. Query them for personal questions unless not applicable. Follow the prompt exactly as it is written. Keep responses short unless explicitly asked otherwise."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
