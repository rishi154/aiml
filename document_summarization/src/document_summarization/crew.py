import os

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from document_summarization.tools.custom_tool import CustomPdfReader
from dotenv import load_dotenv

load_dotenv(".env")

claude_llm = LLM(
	model=os.getenv("MODEL"),
	aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_region_name=os.getenv("AWS_REGION_NAME")
)

@CrewBase
class DocumentSummarization():
	"""DocumentSummarization crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def document_summarizer(self) -> Agent:
		return Agent(
			config=self.agents_config['document_summarizer'],
			verbose=True,
			tools=[CustomPdfReader()],
			llm=claude_llm
		)

	@task
	def summarize(self) -> Task:
		return Task(
			config=self.tasks_config['summarize'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the DocumentSummarization crew"""

		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
		)
