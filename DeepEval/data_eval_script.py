import os
import pandas as pd
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import TaskCompletionMetric, GEval, AnswerRelevancyMetric, FaithfulnessMetric

# 🚨 Ensure your valid OpenAI API key is configured here
#os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# ==========================================
# 1. LOAD YOUR DATASET
# ==========================================
# Assuming you have a CSV file named 'agent_results.csv' with columns:
# 'query', 'response', 'ground_truth', and 'context'
# df = pd.read_csv("agent_results.csv")
# mock_dataset = df.to_dict(orient="records")

# For the sake of this script, here is a mock dataset structure mimicking a loaded file:
mock_dataset = [
    {
        "query": "Fetch the Q3 electrical bus maintenance log and summarize recurring failure modes.",
        "response": "Cooling system leakage occurred in 20 electric busses.",
        "ground_truth": "The Q3 log shows cooling system leakage as the primary failure mode, appearing in 12 events.",
        "context": "Cooling system leakage occurred in 12 out of 28 maintenance events during Q3."
    },
    {
        "query": "What is the status of the hydraulic pump replacement on bus 42?",
        "response": "The hydraulic pump on bus 42 was successfully replaced on October 12th.",
        "ground_truth": "Bus 42 had its hydraulic pump replaced on October 12th and is back in service.",
        "context": "Maintenance log: Bus 42 - Hydraulic pump replacement completed Oct 12. Cleared for route."
    }
    # ... Imagine your full 100 rows here
]

# ==========================================
# 2. BUILD THE TEST CASES
# ==========================================
test_cases = []

# Iterate over your dataset and build an LLMTestCase for each row
for row in mock_dataset:
    test_case = LLMTestCase(
        input=row["query"],
        actual_output=row["response"],
        expected_output=row["ground_truth"], 
        # Note: Faithfulness requires retrieval context as a list of strings
        retrieval_context=[row["context"]] 
    )
    test_cases.append(test_case)

print(f"Successfully loaded {len(test_cases)} test cases.")

# ==========================================
# 3. INSTANTIATE YOUR METRICS
# ==========================================
task_completion = TaskCompletionMetric(threshold=0.7)
answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
faithfulness = FaithfulnessMetric(threshold=0.7)

step_efficiency_geval = GEval(
    name="Agent Step Efficiency",
    criteria="Determine if the agent reached the solution directly and efficiently without taking redundant, circular, or unnecessary tool execution steps.",
    # Updated to use SingleTurnParams to fix the deprecation warning
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
    evaluation_steps=[
        "Examine the input request to understand what details the user needed.",
        "Examine the actual output and context to see if the final response represents a direct, efficient answer.",
        "Penalize the score if the agent seems to have hallucinated metrics.",
        "Score 1.0 if the answer is concise and derived directly. Score 0.0 if it indicates an unstable processing path."
    ],
    threshold=0.5
)

# Bundle the metrics into a list
metrics_to_run = [
    task_completion, 
    step_efficiency_geval, 
    answer_relevancy, 
    faithfulness
]

# ==========================================
# 4. RUN BULK EVALUATION
# ==========================================
print("\n==================================================")
print("STARTING BULK DATASET EVALUATION")
print("==================================================\n")

# Removed the print_results argument to fix the TypeError crash
dataset_results = evaluate(
    test_cases=test_cases,
    metrics=metrics_to_run
)

print("\n==================================================")
print("EVALUATION RUN COMPLETE")
print("==================================================")