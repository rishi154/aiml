import os

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import streamlit as st
from dotenv import load_dotenv

from tools.custom_tool import CreditCardNumberGenerator

load_dotenv(".env")

claude_llm = LLM(
	model=os.getenv("MODEL"),
	aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_region_name=os.getenv("AWS_REGION_NAME")
)

@CrewBase
class Crewaichat():
	"""Crewaichat crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def super_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['super_agent'],
			verbose=True,
			tools=[CreditCardNumberGenerator()],
			llm=claude_llm
		)

	@task
	def super_task(self) -> Task:
		return Task(
			config=self.tasks_config['super_task'],
		)

	@crew
	def create_crew(self) -> Crew:
		"""Creates the Crewaichat crew"""
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
		)

	def main(self):
		st.title("Chatbot using Streamlit and CrewaI")
		st.write("Type your message to the chatbot and get a response.")

		# Create the CrewAI system (can be re-used if persistent sessions are needed)
		crew = self.create_crew()

		# Streamlit input box for user message
		user_message = st.text_input("Your message:", "")

		if user_message:
			# Run the crew (chatbot agent responds)
			response = crew.kickoff(inputs={"input": user_message})

			# Display the response
			st.write(response)

if __name__ == "__main__":
	Crewaichat().main()