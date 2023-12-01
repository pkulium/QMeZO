#!/bin/bash

# Define arrays of values to try
learning_rates=(1e-6 2e-6)
epsilons=(1e-3 2e-3)

# Loop over all combinations
for lr in "${learning_rates[@]}"; do
    for eps in "${epsilons[@]}"; do
        echo "Running with LR=$lr and EPS=$eps"
        # Run your script here, passing in LR and EPS
        LR=$lr EPS=$eps sh test.sh
    done
done
