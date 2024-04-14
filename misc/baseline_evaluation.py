import argparse
from misc.util import select_best_hps, param_comp
from misc.generate_hp_search_scripts import alg_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_log_dir", type=str, default="cluster_logs")
    parser.add_argument("--learner", type=str, default="sac", choices=["ppo", "sac", "dqn"])
    parser.add_argument("--env", type=str, default="maze",
                        choices=["point_mass_2d", "sparse_goal_reaching", "unlockpickup"])

    args = parser.parse_args()
    if args.env == "point_mass_2d":
        from deep_sprl.experiments import PointMass2DExperiment
        exp = PointMass2DExperiment
    elif args.env == "sparse_goal_reaching":
        from deep_sprl.experiments import SparseGoalReachingExperiment
        exp = SparseGoalReachingExperiment
    elif args.env == "unlockpickup":
        from deep_sprl.experiments import UnlockPickupExperiment
        exp = UnlockPickupExperiment
    else:
        raise RuntimeError("Unknown environment: %s" % args.env)

    for method, params in alg_params.items():
        top_hps, top_perfs = select_best_hps(exp, args.learner, method, {k: v for (k, v) in params}, args.base_log_dir)
        print(top_hps)
        print(top_perfs)
        param_comp(exp, args.learner, teacher=method, full_params=top_hps, log_dir=args.base_log_dir)


if __name__ == "__main__":
    main()
