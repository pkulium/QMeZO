#!/bin/bash


# Define the list of tasks
# TASKS=("SST2" "RTE" "CB" "BoolQ" "WSC" "WIC" "MultiRC" "Copa" "ReCoRD" "SQuAD" "DROP")
TASKS=("RTE"  "WIC" "Copa" "ReCoRD" "SQuAD")

# Loop over each task
for TASK in "${TASKS[@]}"; do
   echo "Running script for TASK: $TASK"

   # Call your existing script here, passing TASK as an argument
   # Assuming your script is named 'run_training.sh'
   TASK=$TASK sh ./mezo.sh
done
