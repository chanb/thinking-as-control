d_model=16
n_heads=1

mkdir -p EXPS
mkdir -p EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}
mkdir -p EXPS/logs-d_model_${d_model}-n_heads_${n_heads}

### NO THINKING
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=0 --seed=42 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-no_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-no_thought_acts_42.log 2>&1 &
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=0 --seed=43 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-no_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-no_thought_acts_43.log 2>&1 &
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=0 --seed=44 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-no_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-no_thought_acts_44.log 2>&1 &
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=0 --seed=45 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-no_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-no_thought_acts_45.log 2>&1 &
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=0 --seed=46 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-no_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-no_thought_acts_46.log 2>&1 &

### THINKING x3
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=3 --seed=42 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-3_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-3_thought_acts_42.log 2>&1 &
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=3 --seed=43 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-3_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-3_thought_acts_43.log 2>&1 &
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=3 --seed=44 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-3_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-3_thought_acts_44.log 2>&1 &
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=3 --seed=45 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-3_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-3_thought_acts_45.log 2>&1 &
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=3 --seed=46 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-3_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-3_thought_acts_46.log 2>&1 &

### THINKING x10
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=10 --seed=42 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-10_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-10_thought_acts_42.log 2>&1 &
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=10 --seed=43 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-10_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-10_thought_acts_43.log 2>&1 &
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=10 --seed=44 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-10_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-10_thought_acts_44.log 2>&1 &
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=10 --seed=45 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-10_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-10_thought_acts_45.log 2>&1 &
python thought_gridworld-rl.py --env=v1 --d_model=${d_model} --n_thought_states=0 --n_thought_acts=10 --seed=46 --output_file=EXPS/debug_results-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-10_thought_acts > EXPS/logs-d_model_${d_model}-n_heads_${n_heads}/test_thought_env-10_thought_acts_46.log 2>&1 &

wait