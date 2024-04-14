#!/bin/bash
LOG_DIR="logs_her"
for SEED in $(seq $1 $2)
do
  # For default sampling we already have the data from the main experiment, hence only the HER experiment
  python run.py --base_log_dir $LOG_DIR --type default --learner sac --env sparse_goal_reaching --HER_REPLAYS 2 --ONLINE_HER True --seed $SEED
  # For random sampling we already have the data from the main experiment, hence only the HER experiment. Note how
  # online sampling of HER transitions lead to worse performance compared to offline sampling. For each method we
  # evaluated both during HP search and chose the better performing one.
  python run.py --base_log_dir $LOG_DIR --type random --learner sac --env sparse_goal_reaching --HER_REPLAYS 2 --ONLINE_HER False --seed $SEED
  # For random sampling of target tolerance contexts, we need to also run the default experiments
  python run.py --base_log_dir $LOG_DIR --type random --learner sac --env sparse_goal_reaching --TARGET_SLICE True --seed $SEED
  python run.py --base_log_dir $LOG_DIR --type random --learner sac --env sparse_goal_reaching --TARGET_SLICE True --HER_REPLAYS 2 --ONLINE_HER True --seed $SEED
  # For CurrOT and Gradient, we again need to only run the HER experiments and can re-use the SGR results from the main
  # paper
  python run.py --base_log_dir $LOG_DIR --type wasserstein --learner sac --env sparse_goal_reaching --HER_REPLAYS 2 --ONLINE_HER False --seed $SEED
  python run.py --base_log_dir $LOG_DIR --type gradient --learner sac --env sparse_goal_reaching --HER_REPLAYS 2 --ONLINE_HER False --seed $SEED
done
