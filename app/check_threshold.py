import os
import sys
import mlflow

THRESHOLD = 0.85

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking accuracy for Run ID: {run_id}")

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

if accuracy is None:
    print("error: no accuracy")
    sys.exit(1)

print(f"accuracy : {accuracy}")
print(f"Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print("failed: accuracy is below threshold")
    sys.exit(1)

print("passed: accuracy meets threshold, start deployment")