import streamlit as st
from basecode import get_chain,create_vector_db
from main2 import predictor

st.title(" Ask me any Questions")

question = st.text_input("question: ")
st.button("enter")

""""
if question:
    chain = get_chain()
    response = chain(question)

    st.header("answer")
    st.write(response["result"])

"""
if predictor(question)=="not stress":
    print("ok I think you are not in stress")

else :
    chain = get_chain()
    response = chain(question)

    st.header("answer")
    st.write(response["result"])
