#!/bin/bash
for SEED in $(seq $1 $2)
do
  python run.py --type self_paced --learner ppo --env point_mass_2d --seed $SEED
  python run.py --type wasserstein --learner ppo --env point_mass_2d --seed $SEED
  python run.py --type gradient --learner ppo --env point_mass_2d --seed $SEED

  python run.py --type wasserstein --learner sac --env sparse_goal_reaching --seed $SEED
  python run.py --type self_paced --learner sac --env sparse_goal_reaching --seed $SEED
  python run.py --type gradient --learner sac --env sparse_goal_reaching --seed $SEED

  python run.py --type wasserstein --learner dqn --env unlockpickup --seed $SEED
  python run.py --type gradient --learner dqn --env unlockpickup --seed $SEED

  python run.py --type wasserstein --learner ppo --env emaze --dist_fn shortest_path --seed $SEED
  python run.py --type gradient --learner ppo --env emaze --dist_fn shortest_path --seed $SEED
  python run.py --type wasserstein --learner ppo --env emaze --dist_fn opt_perf --seed $SEED
  python run.py --type gradient --learner ppo --env emaze --dist_fn opt_perf --seed $SEED
  python run.py --type wasserstein --learner ppo --env emaze --dist_fn cur_perf --seed $SEED
  python run.py --type gradient --learner ppo --env emaze --dist_fn cur_perf --seed $SEED
  python run.py --type wasserstein --learner ppo --env emaze --dist_fn euclidean --seed $SEED
  python run.py --type gradient --learner ppo --env emaze --dist_fn euclidean --seed $SEED
done
