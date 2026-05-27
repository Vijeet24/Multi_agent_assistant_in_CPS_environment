#!/usr/bin/env python
import os
import warnings
from single_agent.crew import IcpsDataAnalysisCrew

# Ignore format warnings from internal dependencies
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the ICPS Data Analysis crew.
    """
    
    # -------------------------------------------------------------------------
    # ADD YOUR QUESTION HERE:
    # Simply edit the text inside the triple quotes below for your new query.
    # -------------------------------------------------------------------------
    my_question = """
    Please plot a bar chart of the total values of the order in CAD.
    
    """

    # The key 'assignment' must match the variable name used in your tasks.yaml {assignment}
    inputs = {
        'assignment': my_question.strip()
    }
    
    print(f"\n[CrewAI] Starting task execution with query:\n{inputs['assignment']}\n")
    
    # Kick off the execution execution
    result = IcpsDataAnalysisCrew().crew().kickoff(inputs=inputs)
    
    print("\n--- Task Completed Successfully ---")
    print(result.raw)

if __name__ == "__main__":
    run()