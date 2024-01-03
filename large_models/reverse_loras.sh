#!/bin/bash

# Define the list of tasks
# TASKS=('DROP' 'SQuAD' 'ReCoRD' 'Copa' 'MultiRC' 'WIC' 'WSC' 'BoolQ' 'CB' 'RTE' 'SST2')
TASKS=('DROP' 'SQuAD' 'ReCoRD' 'Copa' 'MultiRC')
# Loop over each task
for TASK in "${TASKS[@]}"; do
    echo "Running script for TASK: $TASK"

    # Call your existing script here, passing TASK as an argument
    # Assuming your script is named 'run_training.sh'
    TASK=$TASK sh ./lora.sh
done
