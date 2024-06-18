#!/bin/bash

model="gpt-4o"
max_depth=4  # max_depth=4 means 5 step lookahead
max_steps=5
branching_factor=5
vf_budget=20
agent="search"  # change this to "prompt" to run the baseline without search

result_dir="shopping_gpt4o_som_search"
instruction_path="agent/prompts/jsons/p_som_cot_id_actree_3s.json"

# Define the batch size variable (how many examples to run before resetting the environment)
batch_size=5

# Define the starting and ending indices
start_idx=0
end_idx=$((start_idx + batch_size))
max_idx=466

# Loop until the starting index is less than or equal to max_idx.
while [ $start_idx -le $max_idx ]
do
    bash scripts/reset_shopping.sh
    bash prepare.sh
    python run.py \
        --instruction_path $instruction_path \
        --test_start_idx $start_idx \
        --test_end_idx $end_idx \
        --model $model \
        --agent_type $agent   --max_depth $max_depth  --branching_factor $branching_factor  --vf_budget $vf_budget   \
        --result_dir $result_dir \
        --test_config_base_dir=config_files/vwa/test_shopping \
        --repeating_action_failure_th 5 --viewport_height 2048 --max_obs_length 3840 \
        --action_set_tag som  --observation_type image_som  \
        --top_p 0.95   --temperature 1.0  --max_steps $max_steps

    # Increment the start and end indices by the batch size
    start_idx=$((start_idx + batch_size))
    end_idx=$((end_idx + batch_size))

    # Ensure the end index does not exceed 466 in the final iteration
    if [ $end_idx -gt $max_idx ]; then
        end_idx=$max_idx
    fi
done
