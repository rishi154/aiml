import os

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools.tools.pdf_search_tool.pdf_search_tool import PDFSearchTool
from dotenv import load_dotenv

load_dotenv(".env")

claude_llm = LLM(
	 model=os.getenv("MODEL"),
	 aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
     aws_region_name=os.getenv("AWS_REGION_NAME")
 )

pdf_search_tool = PDFSearchTool(r"C:\Users\Test\Documents\AIML-TEJAS-Volume-2.pdf",
								config=dict(
									llm=dict(
										provider="aws_bedrock",  # or google, openai, anthropic, llama2, ...
										config=dict(
											model=os.getenv("MODEL"),
										),
									),
									embedder=dict(
										provider="aws_bedrock",  # or openai, ollama, ...
										config=dict(
											model="amazon.titan-embed-text-v2:0",
											task_type="retrieval_document",
										),
									),
								)
								)

@CrewBase
class CrewaiChatRag():
	"""CrewaiChatRag crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			verbose=True,
			tools=[pdf_search_tool],
			llm=claude_llm,
		)


	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['answer_customer_question_task'],
			tools=[pdf_search_tool]
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the CrewaiChatRag crew"""

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
