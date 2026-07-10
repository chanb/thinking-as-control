results_dir=EXPS/frozen_lake-v4-ppo-single_policy
logs_dir=EXPS/frozen_lake-v4-ppo-single_policy-logs

num_iterations=2500

mkdir -p ${results_dir}
mkdir -p ${logs_dir}

seed=42
n_thought_states=1
n_thought_acts=2

python thought_gridworld-rl.py \
    --ent_coef=0.02 \
    --num_iterations=${num_iterations} \
    --algo=ppo:reverse_kl \
    --n_thought_states=${n_thought_states} \
    --n_thought_acts=${n_thought_acts} \
    --seed=${seed} \
    --output_file=${results_dir}/n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts} \
    --log_file=${logs_dir}/n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts} \
    --model_save_path=${results_dir}/n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts}