import argparse
from deep_sprl.util.parameter_parser import parse_parameters
import deep_sprl.environments
import torch


def main():
    parser = argparse.ArgumentParser("Self-Paced Learning experiment runner")
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--type", type=str, default="wasserstein",
                        choices=["default", "random", "self_paced", "wasserstein", "alp_gmm",
                                 "goal_gan", "acl", "plr", "vds", "gradient"])
    parser.add_argument("--learner", type=str, default="ppo", choices=["ppo", "sac", "dqn"])
    parser.add_argument("--env", type=str, default="point_mass_2d",
                        choices=["point_mass_2d", "sparse_goal_reaching", "emaze",
                                 "unlockpickup", "point_mass_nd"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_cores", type=int, default=1)

    args, remainder = parser.parse_known_args()
    parameters = parse_parameters(remainder)

    torch.set_num_threads(args.n_cores)

    if args.env == "point_mass_2d":
        from deep_sprl.experiments import PointMass2DExperiment
        exp = PointMass2DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == "point_mass_nd":
        from deep_sprl.experiments import PointMassNDExperiment
        exp = PointMassNDExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == "sparse_goal_reaching":
        from deep_sprl.experiments import SparseGoalReachingExperiment
        exp = SparseGoalReachingExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == "emaze":
        from deep_sprl.experiments import EMazeExperiment
        exp = EMazeExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    elif args.env == "unlockpickup":
        from deep_sprl.experiments import UnlockPickupExperiment
        exp = UnlockPickupExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed)
    else:
        raise RuntimeError("Unknown environment '%s'!" % args.env)

    exp.train()
    exp.evaluate()


if __name__ == "__main__":
    main()
