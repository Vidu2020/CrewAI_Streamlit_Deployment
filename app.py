import streamlit as st
from crewai import Agent, Task, Crew, Process
import os

# 1. UI Setup
st.set_page_config(page_title="CrewAI Researcher", page_icon="🤖")
st.title("🤖 AI Research Assistant (CrewAI)")
st.write("Enter a topic, and our multi-agent team will research and summarize it.")

# 2. Sidebar for API Keys (Crucial for Cloud Deployment)
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("Get an API key at [OpenAI](https://platform.openai.com/api-keys)")

# 3. User Input
topic = st.text_input("Enter a research topic (e.g., 'Quantum Computing'):")

# 4. Agentic AI Logic
if st.button("Start Research"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    elif not topic:
        st.warning("Please enter a topic.")
    else:
        # Set the environment variable temporarily for this run
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize the LLM
                # (Remove the llm = ChatOpenAI(...) line entirely)

        # Create Agents
        with st.spinner("Agents are working... (This may take a minute)"):
            
            researcher = Agent(
                role='Senior Tech Researcher',
                goal=f'Find comprehensive information about {topic}',
                backstory='You are an expert researcher at a top technology firm.',
                verbose=True,
                allow_delegation=False,
                llm="gpt-3.5-turbo"  # Updated
            )

            writer = Agent(
                role='Technical Writer',
                goal=f'Write an easy-to-understand summary about {topic}',
                backstory='You simplify complex topics for general audiences.',
                verbose=True,
                allow_delegation=False,
                llm="gpt-3.5-turbo"  # Updated
            )

            # Create Tasks
            research_task = Task(
                description=f'Research the latest developments in {topic}.',
                expected_output='A detailed list of the top 3 bullet points.',
                agent=researcher
            )

            write_task = Task(
                description=f'Use the research to write a 2-paragraph summary on {topic}.',
                expected_output='A clean, formatted 2-paragraph summary.',
                agent=writer
            )

            # Assemble Crew
            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, write_task],
                process=Process.sequential
            )

            # Execute and Display Results
            try:
                result = crew.kickoff()
                st.success("Research Complete!")
                st.markdown("### Final Report")
                
                # In newer CrewAI versions, result is an object. 
                # We use .raw or str() to get the text.
                st.write(str(result))
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
