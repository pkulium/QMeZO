#!/bin/bash

# Define the list of tasks
# TASKS=("RTE" "CB" "BoolQ" "WSC" "WIC" "MultiRC" "COPA" "ReCoRD" "SQuAD" "DROP")
TASKS=("CB" "BoolQ" "WSC" "WIC" "MultiRC" "COPA" "ReCoRD" "SQuAD" "DROP")


# Loop over each task
for TASK in "${TASKS[@]}"; do
    echo "Running script for TASK: $TASK"

    # Call your existing script here, passing TASK as an argument
    # Assuming your script is named 'run_training.sh'
    TASK=$TASK sh ./lora.sh
done
