### NO THINKING
python thought_gridworld-rl.py --n_thought_acts=0 --seed=42 --output_file=debug_results/test_thought_env-no_thought_acts > logs/test_thought_env-no_thought_acts_42.log 2>&1 &
python thought_gridworld-rl.py --n_thought_acts=0 --seed=43 --output_file=debug_results/test_thought_env-no_thought_acts > logs/test_thought_env-no_thought_acts_43.log 2>&1 &
python thought_gridworld-rl.py --n_thought_acts=0 --seed=44 --output_file=debug_results/test_thought_env-no_thought_acts > logs/test_thought_env-no_thought_acts_44.log 2>&1 &
python thought_gridworld-rl.py --n_thought_acts=0 --seed=45 --output_file=debug_results/test_thought_env-no_thought_acts > logs/test_thought_env-no_thought_acts_45.log 2>&1 &
python thought_gridworld-rl.py --n_thought_acts=0 --seed=46 --output_file=debug_results/test_thought_env-no_thought_acts > logs/test_thought_env-no_thought_acts_46.log 2>&1 &

### THINKING x3
python thought_gridworld-rl.py --n_thought_acts=3 --seed=42 --output_file=debug_results/test_thought_env-3_thought_acts > logs/test_thought_env-3_thought_acts_42.log 2>&1 &
python thought_gridworld-rl.py --n_thought_acts=3 --seed=43 --output_file=debug_results/test_thought_env-3_thought_acts > logs/test_thought_env-3_thought_acts_43.log 2>&1 &
python thought_gridworld-rl.py --n_thought_acts=3 --seed=44 --output_file=debug_results/test_thought_env-3_thought_acts > logs/test_thought_env-3_thought_acts_44.log 2>&1 &
python thought_gridworld-rl.py --n_thought_acts=3 --seed=45 --output_file=debug_results/test_thought_env-3_thought_acts > logs/test_thought_env-3_thought_acts_45.log 2>&1 &
python thought_gridworld-rl.py --n_thought_acts=3 --seed=46 --output_file=debug_results/test_thought_env-3_thought_acts > logs/test_thought_env-3_thought_acts_46.log 2>&1 &

### THINKING x10
python thought_gridworld-rl.py --n_thought_acts=10 --seed=42 --output_file=debug_results/test_thought_env-10_thought_acts > logs/test_thought_env-10_thought_acts_42.log 2>&1 &
python thought_gridworld-rl.py --n_thought_acts=10 --seed=43 --output_file=debug_results/test_thought_env-10_thought_acts > logs/test_thought_env-10_thought_acts_43.log 2>&1 &
python thought_gridworld-rl.py --n_thought_acts=10 --seed=44 --output_file=debug_results/test_thought_env-10_thought_acts > logs/test_thought_env-10_thought_acts_44.log 2>&1 &
python thought_gridworld-rl.py --n_thought_acts=10 --seed=45 --output_file=debug_results/test_thought_env-10_thought_acts > logs/test_thought_env-10_thought_acts_45.log 2>&1 &
python thought_gridworld-rl.py --n_thought_acts=10 --seed=46 --output_file=debug_results/test_thought_env-10_thought_acts > logs/test_thought_env-10_thought_acts_46.log 2>&1 &

wait