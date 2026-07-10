d_model=1
results_dir=EXPS/frozen_lake-v4-ppo-iterations_2500-ent_coef_0.02-d_model_${d_model}
logs_dir=EXPS/frozen_lake-v4-ppo-iterations_2500-ent_coef_0.02-d_model_${d_model}-logs

num_iterations=2500

mkdir -p ${results_dir}
mkdir -p ${logs_dir}

### NO THINKING
for seed in 42 43 44 45 46
# for seed in 42
do
    echo $seed
    for i in 1,1 1,2 1,4 1,8
    do
        IFS=","
        set -- $i
        echo $1 and $2
        n_thought_states=$1
        n_thought_acts=$2

        python thought_gridworld-rl.py --ent_coef=0.02 --d_model=${d_model} --num_iterations=${num_iterations} --algo=ppo:reverse_kl --n_thought_states=${n_thought_states} --n_thought_acts=${n_thought_acts} --seed=${seed} --output_file=${results_dir}/n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts} --log_file=${logs_dir}/n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts} &
    done
    wait
done

for seed in 42 43 44 45 46
# for seed in 42
do
    echo $seed
    python thought_gridworld-rl.py --num_iterations=${num_iterations} --d_model=${d_model} --algo=ppo:reverse_kl --n_thought_states=1 --n_thought_acts=0 --seed=${seed} --output_file=${results_dir}/no_thought_acts --log_file=${logs_dir}/no_thought_acts &
    python thought_gridworld-rl.py --num_iterations=${num_iterations} --d_model=${d_model} --algo=ppo:reverse_kl --n_thought_states=1 --n_thought_acts=0 --thought_per_state --seed=${seed} --output_file=${results_dir}/no_thought_acts-thought_per_state --log_file=${logs_dir}/no_thought_acts-thought_per_state &
    wait
done