### NO THINKING
python gridworld-rl.py --n_thought_acts=0 --num_layers=1 --output_file=results-max_steps_50/test_output-aug_computation-no_thinking-1e-3-wd_0.02-tf_layers_1-fixed_start --seed=42 --model_save_path=test_output-aug_computation-no_thinking-1e-3-wd_0.02-tf_layers_1-fixed_start.pt &

python gridworld-rl.py --n_thought_acts=0 --num_layers=2 --output_file=results-max_steps_50/test_output-aug_computation-no_thinking-1e-3-wd_0.02-tf_layers_2-fixed_start --seed=42 --model_save_path=test_output-aug_computation-no_thinking-1e-3-wd_0.02-tf_layers_2-fixed_start.pt &

python gridworld-rl.py --n_thought_acts=0 --num_layers=8 --output_file=results-max_steps_50/test_output-aug_computation-no_thinking-1e-3-wd_0.02-tf_layers_8-fixed_start --seed=42 --model_save_path=test_output-aug_computation-no_thinking-1e-3-wd_0.02-tf_layers_8-fixed_start.pt &

### THINKING
python gridworld-rl.py --n_thought_acts=3 --num_layers=1 --output_file=results-max_steps_50/test_output-aug_computation-1e-3-wd_0.02-tf_layers_1-fixed_start --seed=42 --model_save_path=test_output-aug_computation-1e-3-wd_0.02-tf_layers_1-fixed_start.pt &

python gridworld-rl.py --n_thought_acts=3 --num_layers=2 --output_file=results-max_steps_50/test_output-aug_computation-1e-3-wd_0.02-tf_layers_2-fixed_start --seed=42 --model_save_path=test_output-aug_computation-1e-3-wd_0.02-tf_layers_2-fixed_start.pt &

### NO THINKING + MARKOV
python gridworld-rl.py --n_thought_acts=0 --num_layers=1 --markov_tf --output_file=results-max_steps_50/test_output-aug_computation-no_thinking-1e-3-wd_0.02-tf_layers_1-markov_tf-fixed_start --seed=42 --model_save_path=test_output-aug_computation-no_thinking-1e-3-wd_0.02-tf_layers_1-markov_tf-fixed_start.pt &

python gridworld-rl.py --n_thought_acts=0 --num_layers=2 --markov_tf --output_file=results-max_steps_50/test_output-aug_computation-no_thinking-1e-3-wd_0.02-tf_layers_2-markov_tf-fixed_start --seed=42 --model_save_path=test_output-aug_computation-no_thinking-1e-3-wd_0.02-tf_layers_2-markov_tf-fixed_start.pt &

python gridworld-rl.py --n_thought_acts=0 --num_layers=1 --markov_tf --output_file=results-max_steps_50/test_output-aug_computation-no_thinking-1e-3-wd_0.02-tf_layers_8-markov_tf-fixed_start --seed=42 --model_save_path=test_output-aug_computation-no_thinking-1e-3-wd_0.02-tf_layers_8-markov_tf-fixed_start.pt &

wait