results_dir=EXPS/frozen_lake-ppo
logs_dir=EXPS/logs-frozen_lake-ppo

mkdir -p ${results_dir}
mkdir -p ${logs_dir}

### NO THINKING
for seed in 42 43 44 45 45
do
    echo $seed
    python thought_gridworld-rl.py --n_thought_states=1 --n_thought_acts=0 --seed=${seed} --output_file=${results_dir}/no_thought_acts > ${logs_dir}/no_thought_acts_${seed}.log 2>&1 &
    python thought_gridworld-rl.py --tabular --n_thought_states=1 --n_thought_acts=0 --seed=${seed} --output_file=${results_dir}/no_thought_acts-tabular > ${logs_dir}/no_thought_acts-tabular_${seed}.log 2>&1 &

    for i in 3,3 3,10 10,3 10,10
    do
        IFS=","
        set -- $i
        echo $1 and $2
        n_thought_states=$1
        n_thought_acts=$2

        python thought_gridworld-rl.py --n_thought_states=${n_thought_states} --n_thought_acts=${n_thought_acts} --seed=${seed} --output_file=${results_dir}/n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts} > ${logs_dir}/n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts}_${seed}.log 2>&1 &
    done
    wait
done