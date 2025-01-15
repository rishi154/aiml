import os

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

load_dotenv(".env")

claude_llm = LLM(
	model=os.getenv("MODEL"),
	aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_region_name=os.getenv("AWS_REGION_NAME")
)


@CrewBase
class Codegen():
	"""Codegen crew"""
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def lead_software_developer(self) -> Agent:
		return Agent(
			config=self.agents_config['lead_software_developer'],
			verbose=True,
			llm=claude_llm
		)

	@agent
	def code_reviewer(self) -> Agent:
		return Agent(
			config=self.agents_config['code_reviewer'],
			verbose=True,
			llm=claude_llm
		)

	@task
	def develop_code(self) -> Task:
		return Task(
			config=self.tasks_config['develop_code'],
		)

	@task
	def review_code(self) -> Task:
		return Task(
			config=self.tasks_config['review_code'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Codegen crew"""

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)
