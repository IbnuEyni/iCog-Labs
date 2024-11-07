import streamlit as st

# Page Config
st.set_page_config(page_title="Ebert", page_icon=":movie_camera:")

from utils import write_message
from agents import generate_response

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm Chatbot!  How can I help you?"},
    ]

# Function to clear the cache
def clear_cache():
    st.cache_data.clear()
    st.success("Cache cleared!")

# Button to clear the cache
if st.button("Clear Cache"):
    clear_cache()

# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'): 
        try:
            response = generate_response(message)
            # from time import sleep
            # sleep(1)
            write_message('assistant', response)
        except Exception as e:
            write_message('assistant', f"An error occurred: {str(e)}")


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', question)
    # Generate a response
    handle_submit(question)
