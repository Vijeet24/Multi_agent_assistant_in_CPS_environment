import os
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import TaskCompletionMetric, GEval, AnswerRelevancyMetric, FaithfulnessMetric

# 🚨 Ensure your valid OpenAI API key is configured here
#os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 1. Define your single test case from your dataset
test_case = LLMTestCase(
    input="Fetch the Q3 electrical bus maintenance log and summarize recurring failure modes.",
    actual_output="Cooling system leakage occurred in 20 electric busses",
    retrieval_context=["Cooling system leakage occurred in 12 maintenance events during Q3."]
)

# 2. Instantiate Native DeepEval Metrics
task_completion = TaskCompletionMetric(threshold=0.7)
answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
faithfulness = FaithfulnessMetric(threshold=0.7)

# 3. Instantiate G-Eval Custom Step Efficiency Metric
step_efficiency_geval = GEval(
    name="Agent Step Efficiency",
    criteria="Determine if the agent reached the solution directly and efficiently without taking redundant, circular, or unnecessary tool execution steps.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
    evaluation_steps=[
        "Examine the input request to understand what details the user needed.",
        "Examine the actual output and context to see if the final response represents a direct, efficient answer.",
        "Penalize the score if the agent seems to have hallucinated metrics (e.g., matching '20 electric busses' against a context of '12 out of 28 maintenance events').",
        "Score 1.0 if the answer is concise and derived directly. Score 0.0 if it indicates an unstable processing path."
    ],
    threshold=0.5
)

# 4. Run Evaluations and Print Results
print("==================================================")
print("RUNNING AGENT FRAMEWORK EVALUATION")
print("==================================================\n")

# --- Metric 1: Task Completion ---
print("Evaluating Task Completion...")
task_completion.measure(test_case)
print(f"Task Completion Score: {task_completion.score}")
print(f"Reasoning: {task_completion.reason}\n")

# --- Metric 2: Step Efficiency (via G-Eval) ---
print("Evaluating Step Efficiency...")
step_efficiency_geval.measure(test_case)
print(f"Step Efficiency Score: {step_efficiency_geval.score}")
print(f"Reasoning: {step_efficiency_geval.reason}\n")

# --- Metric 3: Answer Relevancy ---
print("Evaluating Answer Relevancy...")
answer_relevancy.measure(test_case)
print(f"Answer Relevancy Score: {answer_relevancy.score}")
print(f"Reasoning: {answer_relevancy.reason}\n")

# --- Metric 4: Faithfulness (Groundedness) ---
print("Evaluating Faithfulness...")
faithfulness.measure(test_case)
print(f"Faithfulness Score: {faithfulness.score}")
print(f"Reasoning: {faithfulness.reason}\n")

print("==================================================")
print("EVALUATION RUN COMPLETE")
print("==================================================")