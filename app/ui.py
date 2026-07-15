import streamlit as st

from app.graph import build_graph

st.set_page_config(page_title="Home Assistant", page_icon="🏠")


@st.cache_resource
def get_graph():
    return build_graph()


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Household")
    household_type = st.selectbox(
        "Household type",
        options=["apartment", "house", "rental"],
        index=1,
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("🏠 Home Assistant")

for message in st.session_state.messages:
    role = "user" if message[0] == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message[1])

question = st.chat_input("Ask about an appliance, your insurance, or anything else...")

if question:
    st.session_state.messages.append(("human", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            graph = get_graph()
            result = graph.invoke(
                {
                    "household_type": household_type,
                    "messages": st.session_state.messages,
                }
            )
            answer = result["messages"][-1].content
        st.markdown(answer)

    st.session_state.messages.append(("ai", answer))
