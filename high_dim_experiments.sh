#!/bin/bash
for DIM in $(seq 10 1)
do
  echo "Dimension: $DIM"
  for SEED in $(seq $1 $2)
  do
    # For the min-reduction envs, we ablate the BETTER_INIT for ul but do not do it for uh (since the result is clear)
	  python run.py --type gradient --learner ppo --env point_mass_nd --seed $SEED --DIM $DIM
	  python run.py --type gradient --learner ppo --env point_mass_nd --seed $SEED --DIM $DIM --BETTER_INIT True
	  python run.py --type gradient --learner ppo --env point_mass_nd --seed $SEED --DIM $DIM --MAX True

	  python run.py --type gradient --learner ppo --env point_mass_nd --seed $SEED --TARGET_SLICES True --DIM $DIM --BETTER_INIT True
	  python run.py --type gradient --learner ppo --env point_mass_nd --seed $SEED --TARGET_SLICES True --DIM $DIM --MAX True

    python run.py --type wasserstein --learner ppo --env point_mass_nd --seed $SEED --DIM $DIM
	  python run.py --type wasserstein --learner ppo --env point_mass_nd --seed $SEED --DIM $DIM --BETTER_INIT True
	  python run.py --type wasserstein --learner ppo --env point_mass_nd --seed $SEED --DIM $DIM --MAX True

	  python run.py --type wasserstein --learner ppo --env point_mass_nd --seed $SEED --TARGET_SLICES True --DIM $DIM --BETTER_INIT True
	  python run.py --type wasserstein --learner ppo --env point_mass_nd --seed $SEED --TARGET_SLICES True --DIM $DIM --MAX True
  done
done
