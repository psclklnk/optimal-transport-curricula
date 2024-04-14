for SEED in $(seq $1 $2)
do
  python run.py --type acl --learner ppo --env point_mass_2d --seed $SEED
  python run.py --type plr --learner ppo --env point_mass_2d --seed $SEED
  python run.py --type vds --learner ppo --env point_mass_2d --seed $SEED
  python run.py --type alp_gmm --learner ppo --env point_mass_2d --seed $SEED
  python run.py --type goal_gan --learner ppo --env point_mass_2d --seed $SEED
  python run.py --type default --learner ppo --env point_mass_2d --seed $SEED
  python run.py --type random --learner ppo --env point_mass_2d --seed $SEED

  python run.py --type acl --learner sac --env sparse_goal_reaching --seed $SEED
  python run.py --type plr --learner sac --env sparse_goal_reaching --seed $SEED
  python run.py --type vds --learner sac --env sparse_goal_reaching --seed $SEED
  python run.py --type alp_gmm --learner sac --env sparse_goal_reaching --seed $SEED
  python run.py --type goal_gan --learner sac --env sparse_goal_reaching --seed $SEED
  python run.py --type random --learner sac --env sparse_goal_reaching --seed $SEED
  python run.py --type default --learner sac --env sparse_goal_reaching --seed $SEED

  python run.py --type acl --learner dqn --env unlockpickup --seed $SEED
  python run.py --type plr --learner dqn --env unlockpickup --seed $SEED
  python run.py --type vds --learner dqn --env unlockpickup --seed $SEED
  python run.py --type default --learner dqn --env unlockpickup --seed $SEED
  python run.py --type random --learner dqn --env unlockpickup --seed $SEED

  python run.py --type default --learner ppo --env emaze --dist_fn shortest_path --seed $SEED
  python run.py --type random --learner ppo --env emaze --dist_fn shortest_path --seed $SEED
done
