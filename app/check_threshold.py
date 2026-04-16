import sys

THRESHOLD = 0.70

with open("model_info.txt", "r") as f:
    accuracy = float(f.read().strip())

print(f"Accuracy: {accuracy}")
print(f"Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print("failed: accuracy is below threshold")
    sys.exit(1)

print("passed: accuracy meets threshold, start deployment")