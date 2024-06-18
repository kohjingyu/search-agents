#!/bin/bash
### This script runs the GPT-4V + SoM models on the entire VWA shopping test set.

model="gpt-4o"
vf="gpt4o"
max_depth=4  # max_depth=4 means 5 step lookahead
max_steps=5
branching_factor=5
vf_budget=20
agent="search"

result_dir="shopping_gpt4o_som_search"
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s.json"

# Define the batch size variable (how many examples to run before resetting the environment)
batch_size=5
id_list=(11 12 13 16 19 22 24 25 26 27 28 29 30 37 38 39 47 48)
# Initialize an array to hold batches of IDs
batch=()

# Loop over the list of IDs
for id in "${id_list[@]}"
do
    # Add the current ID to the batch
    batch+=($id)

    # Check if the batch is full
    if [ ${#batch[@]} -eq $batch_size ]; then
        # Join the batch into a comma-separated string
        test_idx=$(IFS=,; echo "${batch[*]}")
        echo "Processing batch: $test_idx"
        
        # Reset the batch
        batch=()
        
        # Run the scripts and the Python command with the current batch of IDs
        bash scripts/reset_shopping.sh
        bash prepare.sh

        python run.py \
            --instruction_path $instruction_path \
            --test_idx $test_idx \
            --model $model \
            --agent_type $agent   --max_depth $max_depth  --branching_factor $branching_factor  --vf_budget $vf_budget   \
            --result_dir $result_dir \
            --test_config_base_dir=config_files/vwa/test_shopping \
            --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
            --observation_type accessibility_tree_with_captioner \
            --top_p 0.95   --temperature 1.0  --max_steps $max_steps
    fi
done


# Process any remaining IDs in the batch
if [ ${#batch[@]} -ne 0 ]; then
    test_idx=$(IFS=,; echo "${batch[*]}")
    
    bash scripts/reset_shopping.sh
    bash prepare.sh
    python run.py \
        --instruction_path $instruction_path \
        --test_idx $test_idx \
        --model $model \
        --agent_type $agent   --max_depth $max_depth  --branching_factor $branching_factor  --vf_budget $vf_budget   \
        --result_dir $result_dir \
        --test_config_base_dir=config_files/vwa/test_shopping \
        --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
        --observation_type accessibility_tree_with_captioner \
        --top_p 0.95   --temperature 1.0  --max_steps $max_steps
fi
