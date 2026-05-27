import sys
import warnings
import os
from datetime import datetime

# Import your crew class from your crew.py file
from multi_agent.crew import IndustrialSystemCrew

# Suppress specific warnings as per your reference
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Create output directory if it doesn't exist to support your task output_file paths
os.makedirs('output', exist_ok=True)

# Define the user query that the manager agent will analyze and delegate
user_query = """
I need to check the TIA portal tag configuration for the main assembly line. 
Also, pull the current operational status of the motor from the live data 
and compare its temperature trends with the historical data from last month.
"""

def run():
    """
    Run the Industrial System Crew.
    """
    inputs = {
        'user_query': user_query.strip()
    }

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting CrewAI execution...")
    print(f"User Query: {inputs['user_query']}\n")

    # Create and run the crew
    try:
        result = IndustrialSystemCrew().crew().kickoff(inputs=inputs)
        
        print("\n================================================")
        print("FINAL MANAGER RESPONSE:")
        print("================================================")
        print(result)
        
    except Exception as e:
        print(f"An error occurred during crew execution: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run()