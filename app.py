import os
import sys

# SQLite workaround for Streamlit Cloud deployment
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import streamlit as st
from crewai import Agent, Task, Crew, Process

st.set_page_config(page_title="CrewAI Streamlit Deployment", page_icon="🤖", layout="centered")

st.title("CrewAI + Streamlit Demo")
st.write("A simple CrewAI app deployed with Streamlit.")

# Ask user for API key in the sidebar and mask the input
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

# Set the environment variable so CrewAI can find it automatically
os.environ["OPENAI_API_KEY"] = api_key

topic = st.text_input("Enter a topic", value="Future of AI in healthcare")

if st.button("Run Crew"):
    with st.spinner("Running crew..."):
        try:
            researcher = Agent(
                role="Research Analyst",
                goal="Provide a concise and useful explanation about the given topic.",
                backstory="You are a helpful research analyst who writes clear summaries.",
                verbose=True
            )

            task = Task(
                description=f"Write a short summary about: {topic}",
                expected_output="A clear, concise summary in plain English.",
                agent=researcher
            )

            crew = Crew(
                agents=[researcher],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )

            result = crew.kickoff()
            st.success("Crew completed successfully.")
            st.subheader("Result")
            st.write(str(result))

        except Exception as e:
            st.exception(e)
