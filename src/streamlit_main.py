import streamlit as st

st.set_page_config(
    page_title = "Multipage App",
    page_icon = " @@ ~\'O'/~"
)

st.title("Main Page")
st.sidebar.success("Select a page above")

if "my_input" not in st.session_state:
    st.session_state["my_input"] = ""

my_input = st.text_input("Input API Key", st.session_state["my_input"])
submit = st.button("Submit")
if submit: 
    st.session_state["my_input"]=my_input
    st.write("API KEY ENTERED")
    my_input.capitalize()
