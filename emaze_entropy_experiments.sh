for SEED in $(seq $1 $2)
do
  for ENT_LB in 0.5 1.0 2.0;
  do
    python run.py --type wasserstein --learner ppo --env emaze --dist_fn shortest_path --ENT_LB $ENT_LB --seed $SEED
    python run.py --type wasserstein --learner ppo --env emaze --dist_fn opt_perf --ENT_LB $ENT_LB --seed $SEED
    python run.py --type wasserstein --learner ppo --env emaze --dist_fn cur_perf --ENT_LB $ENT_LB --seed $SEED
    python run.py --type wasserstein --learner ppo --env emaze --dist_fn euclidean --ENT_LB $ENT_LB --seed $SEED
  done

  for GRADIENT_ENT in 1e-8 1e-4 1e-2;
  do
    python run.py --type gradient --learner ppo --env emaze --dist_fn shortest_path --GRADIENT_ENT $GRADIENT_ENT --seed $SEED
    python run.py --type gradient --learner ppo --env emaze --dist_fn opt_perf --GRADIENT_ENT $GRADIENT_ENT --seed $SEED
    python run.py --type gradient --learner ppo --env emaze --dist_fn cur_perf --GRADIENT_ENT $GRADIENT_ENT --seed $SEED
    python run.py --type gradient --learner ppo --env emaze --dist_fn euclidean --GRADIENT_ENT $GRADIENT_ENT --seed $SEED
  done
done