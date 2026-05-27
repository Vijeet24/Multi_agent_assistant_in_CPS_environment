import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import CSVSearchTool, TXTSearchTool, CodeInterpreterTool

@CrewBase
class IcpsDataAnalysisCrew():
    """ICPS Data Analysis Crew for processing laboratory documents and operational telemetry."""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def lab_analyst(self) -> Agent:
        # Define the exact path to your telemetry/experimental data
        csv_file_path = ".E:\agents\single_agent\csv_data.csv"
        txt_file_path = "E:\agents\single_agent\tia_doc.txt"
        
        # 1. RAG Tools: Used when the query asks to search/find specific text or rows
        csv_search_tool = CSVSearchTool(csv_file_path=csv_file_path)
        txt_search_tool = TXTSearchTool(txt_file_path=txt_file_path)

        # 2. Advanced Coding Tool: Explicitly pass the file to the interpreter
        # This allows the agent's Python environment to find and load the dataset
        code_interpreter = CodeInterpreterTool(workspace_dir="./icps_data")

        return Agent(
            config=self.agents_config['lab_analyst'],
            verbose=True,
            # Giving the agent both search capability and programmatic execution capability
            tools=[csv_search_tool, txt_search_tool, code_interpreter],
            allow_code_execution=True,
            code_execution_mode="safe",  # Runs code inside Docker
            max_execution_time=120,      # Extended time to allow for data processing and chart rendering
            max_retry_limit=3
        )

    @task
    def icps_data_analysis_task(self) -> Task:
        os.makedirs("output", exist_ok=True)
        return Task(
            config=self.tasks_config['icps_data_query_answering_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ICPS Data Analysis crew"""
        return Crew(
            agents=self.agents, 
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
