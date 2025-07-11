import streamlit as st

st.title("Random Forest Streamlit App")

name = st.text_input("What's your name?")
age = st.number_input("How old are you?", min_value=1)

if st.button("Say Hello"):
    st.success(f"Hello {name}, you are {age} years old!")

