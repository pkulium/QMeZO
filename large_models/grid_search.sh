#!/bin/bash

# Define arrays of values to try
learning_rates=(1e-6 5e-6 1e-7 5e-7)
epsilons=(1e-3 5e-3 1e-4 5e-4)

# Loop over all combinations
for lr in "${learning_rates[@]}"; do
    for eps in "${epsilons[@]}"; do
        echo "Running with LR=$lr and EPS=$eps"
        # Run your script here, passing in LR and EPS
        ./your_script.sh $lr $eps
    done
done
