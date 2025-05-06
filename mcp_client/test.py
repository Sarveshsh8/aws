import streamlit as st
from praisonaiagents import Agent, MCP

#hey

#hajdiq

st.title("🏠 Airbnb Booking Assistant")

# Create the agent
@st.cache_resource
def get_agent():
    return Agent(
        instructions="""You help book apartments on Airbnb.""",
        llm="gpt-4o-mini",  
        tools=MCP("npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt")
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input form
with st.form("booking_form"):
    st.subheader("Enter your booking details")

    destination = st.text_input("Destination:", "Paris")

    col1, col2 = st.columns(2)
    with col1:
        check_in = st.date_input("Check-in date")
    with col2:
        check_out = st.date_input("Check-out date")

    adults = st.number_input("Number of adults:", min_value=1, max_value=10, value=2)

    submitted = st.form_submit_button("Search for accommodations")

    if submitted:
        search_agent = get_agent()

        # Format the query
        query = f"I want to book an apartment in {destination} from {check_in.strftime('%m/%d/%Y')} to {check_out.strftime('%m/%d/%Y')} for {adults} adults"

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})

        # Display user message
        with st.chat_message("user"):
            st.markdown(query)

        # Get response from the agent
        with st.chat_message("assistant"):
            with st.spinner("Searching for accommodations..."):
                response = search_agent.start(query)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Allow for follow-up questions
if st.session_state.messages:
    prompt = st.chat_input("Ask a follow-up question about the accommodations")
    if prompt:
        search_agent = get_agent()

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = search_agent.start(prompt)
                st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response}) 