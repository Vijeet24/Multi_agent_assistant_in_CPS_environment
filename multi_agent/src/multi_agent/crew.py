import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool
from langchain_openai import ChatOpenAI

# ==========================================
# TOOL INITIALIZATION
# ==========================================
# Explicitly pointing to your data files. 
# No dummy data generators are used here.
connection_docs_tool = FileReadTool(file_path='E:\agents\multi_agent\tia_doc.txt')
software_docs_tool = FileReadTool(file_path='E:\agents\multi_agent\tia_doc.txt')
maintenance_docs_tool = FileReadTool(file_path='E:\agents\multi_agent\tia_doc.txt')
current_data_tool = FileReadTool(file_path='E:\agents\multi_agent\csv_data.csv')
historical_data_tool = FileReadTool(file_path='E:\agents\multi_agent\csv_data.csv')


@CrewBase
class IndustrialSystemCrew():
    """Industrial System Query and Support Crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # ==========================================
    # AGENTS
    # ==========================================

    @agent
    def agent_1_connection_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['agent_1_connection_specialist'],
            tools=[connection_docs_tool],
            verbose=True,
        )

    @agent
    def agent_2_software_operations_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['agent_2_software_operations_specialist'],
            tools=[software_docs_tool],
            verbose=True,
        )

    @agent
    def agent_3_maintenance_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['agent_3_maintenance_specialist'],
            tools=[maintenance_docs_tool],
            verbose=True,
        )
    
    @agent
    def agent_4_realtime_status_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['agent_4_realtime_status_analyst'],
            tools=[current_data_tool],
            verbose=True,
            allow_code_execution=True, 
            code_execution_mode="safe", 
            max_execution_time=500, 
            max_retry_limit=3
        )

    @agent
    def agent_5_historical_analysis_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['agent_5_historical_analysis_specialist'],
            tools=[historical_data_tool],
            verbose=True,
            allow_code_execution=True, 
            code_execution_mode="safe", 
            max_execution_time=500, 
            max_retry_limit=3 
        )

    @agent
    def agent_6_visualization_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['agent_6_visualization_specialist'],
            tools=[historical_data_tool],
            verbose=True,
            allow_code_execution=True, 
            code_execution_mode="safe", 
            max_execution_time=500, 
            max_retry_limit=3 
        )

    # ==========================================
    # TASKS
    # ==========================================

    @task
    def connection_setup_task(self) -> Task:
        return Task(config=self.tasks_config['connection_setup_task'])

    @task
    def software_operations_task(self) -> Task:
        return Task(config=self.tasks_config['software_operations_task'])

    @task
    def maintenance_task(self) -> Task:
        return Task(config=self.tasks_config['maintenance_task'])

    @task
    def realtime_monitoring_task(self) -> Task:
        return Task(config=self.tasks_config['realtime_monitoring_task'])

    @task
    def historical_analysis_task(self) -> Task:
        return Task(config=self.tasks_config['historical_analysis_task'])

    @task
    def visualization_task(self) -> Task:
        return Task(config=self.tasks_config['visualization_task'])

    # ==========================================
    # CREW ORCHESTRATION
    # ==========================================

    @crew
    def crew(self) -> Crew:
        """Creates the Industrial System crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical, 
            manager_llm=ChatOpenAI(temperature=0, model="gpt-4o"), 
            verbose=True,
        )

# ==========================================
# EXECUTION ENTRY POINT
# ==========================================
def run():
    """
    Run the crew.
    """
    inputs = {
        'user_query': 'What is the current operational status of the primary cooling pump, and how does its vibration trend compare to last week?'
    }
    IndustrialSystemCrew().crew().kickoff(inputs=inputs)

if __name__ == '__main__':
    run()