import json

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from langchain.chains import RetrievalQA

from src.utils.crewai_translation_rag_tool import FAISSRAGTool

@CrewBase
class TradProject():
	"""TestProject crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	def __init__(self, my_llm):
		self.my_llm = my_llm

	@agent
	def translator(self) -> Agent:
		return Agent(
			config=self.agents_config['translator'],
			verbose=True,
			llm=self.my_llm
		)

	@agent
	def reviewer(self) -> Agent:
		return Agent(
			config=self.agents_config['reviewer'],
			verbose=True,
			llm=self.my_llm
		)


	@task
	def translation_task(self) -> Task:
		return Task(
			config=self.tasks_config['translation_task'],
		)

	@task
	def review_task(self) -> Task:
		return Task(
			config=self.tasks_config['review_task'],
			# output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the TestProject crew"""


		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical,
		)


@CrewBase
class RAGTradProject:
    """Translation Project with RAG capabilities"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    def __init__(self, my_llm, json_path):
        self.my_llm = my_llm
        self.rag_tool = FAISSRAGTool(json_path=json_path)


    @agent
    def translator(self) -> Agent:
        return Agent(
            config=self.agents_config['translator'],
            verbose=True,
            llm=self.my_llm,
            tools=[self.rag_tool.query_translation]
        )

    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config['reviewer'],
            verbose=True,
            llm=self.my_llm,
            # tools=[self.rag_tool.query_translation]
        )


    @task
    def translation_task(self) -> Task:
        return Task(
			config=self.tasks_config['translation_task_rag'],
		)

    @task
    def review_task(self) -> Task:
        return Task(
			config=self.tasks_config['review_task_rag'],
			# output_file='report.md'
		)

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
    

if __name__ == "__main__":
	llm = LLM(
		model="openai/llama3.1",
		base_url="http://localhost:1234/v1",
		api_key="asdf"
	)
	DATA_PATH = '/Users/fvelasco/data/research/mt-agents-research/data/trad_doc_store/'
	# Load your translations
	# with open(f'{DATA_PATH}translations.json', 'r') as f:
	# 	translations = json.load(f)['translations']

	project = RAGTradProject(llm, json_path=f'{DATA_PATH}translations.json')

	# project = TradProject(llm)
	text = "Hello, my name is Iñigo Montoya. You killed my father. Prepare to die"
	# Run the crew
	result = project.crew().kickoff(inputs={'source_lang': 'en', 
                                                     'target_lang': 'es', 
                                                     'segment': text})
