import os

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from synthetic_data_generator.tools.custom_tool import CreditCardNumberGenerator

load_dotenv(".env")

claude_llm = LLM(
	model=os.getenv("MODEL"),
	aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_region_name=os.getenv("AWS_REGION_NAME")
)

@CrewBase
class SyntheticDataGenerator():
	"""SyntheticDataGenerator crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def synthetic_data_generator(self) -> Agent:
		return Agent(
			config=self.agents_config['synthetic_data_generator'],
			verbose=True,
			tools=[CreditCardNumberGenerator()],
			llm=claude_llm
		)

	@task
	def generate_data(self) -> Task:
		return Task(
			config=self.tasks_config['generate_data'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the SyntheticDataGenerator crew"""
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
		)
