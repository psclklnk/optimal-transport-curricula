This folder contains the code accompanying the T-PAMI paper "On the Benefit of Optimal Transport for Curriculum Reinforcement Learning". 

For the sparse goal reaching, unlockpickup, and point mass environment, we used Python 3.8 to run the experiments. The required dependencies are 
listed in the **requirements.txt** file and can be installed via
```shell script
cd nadaraya-watson
pip install .
pip install -r requirements.txt
```
Note that on a Macbook, you may need to run 
```shell script
export OpenMP_ROOT=$(brew --prefix)/opt/libomp 
```
if you have installed OpenMP via Homebrew and CMake is unable to find the OpenMP installation.

The experiments can be run via the **run.py** scripts. For convenience, we have created the **baseline_experiments.sh** 
and **interpolation_based_experiments.sh** scripts for running the baselines and interpolation-based algorithms.
To execute all experiments for seeds 1 - 2, you can run
```shell script
./baseline_experiments.sh 1 2
./interpolation_based_experiments.sh 1 2
./emaze_entropy_experiments.sh 1 2
./high_dim_experiments.sh 1 2
./sgr_her_experiments.sh 1 2
```
After running the desired amount of seeds, you can visualize the results via
```shell script
cd misc
./visualize_results.sh
```
Note that if you decide to run only some of the experiments from the scripts, you may need to adjust the python
scripts for visualization and then shell script for visualization.

## UnlockPickup

For discrete environments, CurrOT pre-computes the neighbours in the $\epsilon$-ball around each context such
that we do not need to re-compute this information online. It stores the pre-computed map on disk. Consequently,
it is wise to first compute this map by running 
`````shell
python run.py --type wasserstein --learner dqn --env unlockpickup --seed 1
`````
and once the pre-computation of the map is done, start other seeds as desired. Otherwise, all seeds will compute
the neighbour map upon first executing.

## TeachMyAgent

In order to run the TeachMyAgent experiments, you will first need to clone the code from the 
accompanying [Github repository](https://github.com/flowersteam/TeachMyAgent) and then copy the
following files

| Source | Destination |
|--------|-------------|
| `deep_sprl/teachers/spl/currot_tma.py` | `TeachMyAgent/teachers/algos/currot_tma.py` |
| `deep_sprl/teachers/spl/currot.py` | `TeachMyAgent/teachers/algos/currot.py` |
| `deep_sprl/teachers/spl/currot_utils.py` | `TeachMyAgent/teachers/algos/currot_internal/currot_utils.py` |
| `deep_sprl/teachers/spl/wasserstein_interpolation.py` | `TeachMyAgent/teachers/algos/currot_internal/wasserstein_interpolation.py` |
| `deep_sprl/teachers/util.py` | `TeachMyAgent/teachers/algos/currot_internal/util.py` |

It will be necessary to fix the import statements in the files and additionally remove the **AbstractTeacher** parent class
from **CurrOT** and **Gradient** in the **currot.py** file. Required libraries like the nadaraya-watson module also need
to be installed in the conda- or virtual environment of TeachMyAgent.

Additionally, the **teacher_args_handler.py** file from the TeachMyAgent repository needs to be modified. We need
to add the CurrOT and Gradient arguments
```python
parser.add_argument("--perf_lb", type=float, default=None)
parser.add_argument("--n_samples", type=int, default=500)
parser.add_argument("--episodes_per_update", type=int, default=50)
parser.add_argument("--epsilon", type=float, default=None)
parser.add_argument("--optimize_initial_samples", action="store_true")
```
in the **set_parser_arguments** method and also add code for processing the arguments in the **get_object_from_arguments**
method
```python
if args.teacher == 'WB-SPRL':
    params["perf_lb"] = args.perf_lb
    params["n_samples"] = args.n_samples
    params["epsilon"] = args.epsilon
    params["episodes_per_update"] = args.episodes_per_update

if args.teacher == 'Gradient':
    params["perf_lb"] = args.perf_lb
    params["n_samples"] = args.n_samples
    params["epsilon"] = args.epsilon
    params["episodes_per_update"] = args.episodes_per_update
    params["optimize_initial_samples"] = args.optimize_initial_samples
```

Finally, we modify the **teacher_controller.py** file by first importing the Currot and Gradient interfaces
```python
from TeachMyAgent.teachers.algos.currot_tma import TMACurrot, TMAGradient
```
and then adding the following additional clauses
```python
elif teacher == 'WB-SPRL':
    self.task_generator = TMACurrot(mins, maxs, seed=seed, **teacher_params)
elif teacher == 'Gradient':
    self.task_generator = TMAGradient(mins, maxs, seed=seed, **teacher_params)
```
in the constructor of the **TeacherController** class. The $`\delta`$ and $`\epsilon`$ values detailed in the appendix 
can then be passed to the algorithms via the `perf_lb` and `epsilon` arguments.
