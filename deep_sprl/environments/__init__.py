from gym.envs.registration import register

register(
    id='ContextualPointMass2D-v1',
    max_episode_steps=100,
    entry_point='deep_sprl.environments.contextual_point_mass_2d:ContextualPointMass2D'
)

register(
    id='HighDimPointMass-v1',
    max_episode_steps=100,
    entry_point='deep_sprl.environments.high_dim_point_mass:HighDimPointMass'
)

register(
    id='SparseGoalReaching-v1',
    max_episode_steps=200,
    entry_point='deep_sprl.environments.sparse_goal_reaching:SparseGoalReachingEnv'
)

register(
    id='EMaze-v1',
    max_episode_steps=150,
    entry_point='deep_sprl.environments.emaze:EMazeEnv'
)

register(
    id='UnlockPickup-v1',
    entry_point='deep_sprl.environments.unlockpickup.unlock_pickup_env:ContextualUnlockPickupEnv'
)
