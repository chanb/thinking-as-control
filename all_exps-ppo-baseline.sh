results_dir=EXPS/frozen_lake-v4-ppo-iterations_2500-ent_coef_0.02
logs_dir=EXPS/frozen_lake-v4-ppo-iterations_2500-ent_coef_0.02-logs

num_iterations=2500

mkdir -p ${results_dir}
mkdir -p ${logs_dir}

### NO THINKING
for seed in 42 43 44 45 46
# for seed in 42
do
    echo $seed
    python thought_gridworld-rl.py --ent_coef=0.02 --num_iterations=${num_iterations} --algo=ppo:reverse_kl --n_thought_states=1 --n_thought_acts=0 --seed=${seed} --output_file=${results_dir}/no_thought_acts --log_file=${logs_dir}/no_thought_acts
done