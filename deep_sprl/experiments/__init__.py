# Do this import to ensure that the Gym environments get registered properly
import deep_sprl.environments
from .point_mass_2d_experiment import PointMass2DExperiment
from .point_mass_nd_experiment import PointMassNDExperiment
from .sparse_goal_reaching_experiment import SparseGoalReachingExperiment
from .emaze_experiment import EMazeExperiment
from .unlockpickup_experiment import UnlockPickupExperiment
from .abstract_experiment import CurriculumType, Learner

__all__ = ['CurriculumType', 'PointMass2DExperiment', 'PointMassNDExperiment', 'SparseGoalReachingExperiment',
           'Learner', 'EMazeExperiment', 'UnlockPickupExperiment']
