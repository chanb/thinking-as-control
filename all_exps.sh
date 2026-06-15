env=v3

results_dir=EXPS/env_${env}-ppo
logs_dir=EXPS/logs-env_${env}-ppo

mkdir -p ${results_dir}
mkdir -p ${logs_dir}

### NO THINKING
python thought_gridworld-rl.py --env=${env} --n_thought_states=1 --n_thought_acts=0 --seed=42 --output_file=${results_dir}/test_thought_env-no_thought_acts > ${logs_dir}/test_thought_env-no_thought_acts_42.log 2>&1 &
python thought_gridworld-rl.py --env=${env} --n_thought_states=1 --n_thought_acts=0 --seed=43 --output_file=${results_dir}/test_thought_env-no_thought_acts > ${logs_dir}/test_thought_env-no_thought_acts_43.log 2>&1 &
python thought_gridworld-rl.py --env=${env} --n_thought_states=1 --n_thought_acts=0 --seed=44 --output_file=${results_dir}/test_thought_env-no_thought_acts > ${logs_dir}/test_thought_env-no_thought_acts_44.log 2>&1 &
python thought_gridworld-rl.py --env=${env} --n_thought_states=1 --n_thought_acts=0 --seed=45 --output_file=${results_dir}/test_thought_env-no_thought_acts > ${logs_dir}/test_thought_env-no_thought_acts_45.log 2>&1 &
python thought_gridworld-rl.py --env=${env} --n_thought_states=1 --n_thought_acts=0 --seed=46 --output_file=${results_dir}/test_thought_env-no_thought_acts > ${logs_dir}/test_thought_env-no_thought_acts_46.log 2>&1 &
wait

for i in 3,3 3,10 10,3 10,10
do
    IFS=","
    set -- $i
    echo $1 and $2
    n_thought_states=$1
    n_thought_acts=$2

    python thought_gridworld-rl.py --env=${env} --n_thought_states=${n_thought_states} --n_thought_acts=${n_thought_acts} --seed=42 --output_file=${results_dir}/test_thought_env-n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts} > ${logs_dir}/test_thought_env-n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts}_42.log 2>&1 &
    python thought_gridworld-rl.py --env=${env} --n_thought_states=${n_thought_states} --n_thought_acts=${n_thought_acts} --seed=43 --output_file=${results_dir}/test_thought_env-n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts} > ${logs_dir}/test_thought_env-n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts}_43.log 2>&1 &
    python thought_gridworld-rl.py --env=${env} --n_thought_states=${n_thought_states} --n_thought_acts=${n_thought_acts} --seed=44 --output_file=${results_dir}/test_thought_env-n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts} > ${logs_dir}/test_thought_env-n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts}_44.log 2>&1 &
    python thought_gridworld-rl.py --env=${env} --n_thought_states=${n_thought_states} --n_thought_acts=${n_thought_acts} --seed=45 --output_file=${results_dir}/test_thought_env-n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts} > ${logs_dir}/test_thought_env-n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts}_45.log 2>&1 &
    python thought_gridworld-rl.py --env=${env} --n_thought_states=${n_thought_states} --n_thought_acts=${n_thought_acts} --seed=46 --output_file=${results_dir}/test_thought_env-n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts} > ${logs_dir}/test_thought_env-n_thought_states_${n_thought_states}-n_thought_acts_${n_thought_acts}_46.log 2>&1 &

    wait
done