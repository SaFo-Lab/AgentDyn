#!/bin/bash
log_dir="logs/update"

mkdir -p $log_dir

#model="gpt-4o-mini-2024-07-18"
#model="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
# model="gpt-4.1-2025-04-14"
# model="gpt-4o-2024-08-06"
# model="claude-3-7-sonnet-20250219"
# model="facebook/Meta-SecAlign-70B" # vllm serve meta-llama/Llama-3.3-70B-Instruct   --tokenizer facebook/Meta-SecAlign-70B   --tensor-parallel-size 4   --enable-lora   --max-lora-rank 64   --lora-modules facebook/Meta-SecAlign-70B=facebook/Meta-SecAlign-70B   --gpu_memory_utilization 0.95
# model="claude-sonnet-4-20250514"
# model="gpt-4.1-2025-04-14"
# model="gemini-2.5-flash"
# model="qwen.qwen3-coder-480b-a35b-instruct"
model="moonshotai/kimi-k2.5"

# export SECAGENT_POLICY_MODEL="gpt-4o-mini-2024-07-18"
#export SECAGENT_POLICY_MODEL="us.anthropic.claude-3-5-sonnet-20241022-v2:0"
# export SECAGENT_POLICY_MODEL="gpt-4.1-2025-04-14"
# export SECAGENT_POLICY_MODEL="gpt-4o-2024-08-06"
# export SECAGENT_POLICY_MODEL="claude-sonnet-4-20250514"
# export SECAGENT_POLICY_MODEL="gpt-4.1-2025-04-14"
# export SECAGENT_POLICY_MODEL="gemini-2.5-flash"
# export SECAGENT_POLICY_MODEL="qwen.qwen3-coder-480b-a35b-instruct"
export SECAGENT_POLICY_MODEL="moonshotai/kimi-k2.5"


# manual policies
# export SECAGENT_GENERATE="False"

# auto policies
export SECAGENT_UPDATE="True"
export SECAGENT_IGNORE_UPDATE_ERROR="True"

export COLUMNS=300

SUITE="dailylife"

SECAGENT_SUITE=$SUITE nohup python -m agentdojo.scripts.benchmark -s $SUITE --model $model --logdir $log_dir > $log_dir/$SECAGENT_POLICY_MODEL-$SUITE-no-attack.log 2>&1 &


SECAGENT_SUITE=$SUITE nohup python -m agentdojo.scripts.benchmark -s $SUITE --model $model --attack important_instructions --logdir $log_dir > $log_dir/$SECAGENT_POLICY_MODEL-$SUITE-attack.log 2>&1 &

# SECAGENT_SUITE=$travel nohup python -m agentdojo.scripts.benchmark -s "travel" --model $model --attack important_instructions --logdir $log_dir > $log_dir/$SECAGENT_POLICY_MODEL-travel-attack.log 2>&1 &

# SECAGENT_SUITE=$workspace nohup python -m agentdojo.scripts.benchmark -s "workspace" --model $model --attack important_instructions --logdir $log_dir > $log_dir/$SECAGENT_POLICY_MODEL-workspace-attack.log 2>&1 &


echo "started"
