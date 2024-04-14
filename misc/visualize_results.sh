#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$PWD/..

python pull_figure.py
python paper_figures.py
python visualize_e_maze_results.py
python visualize_unlockpickup_results.py
python point_mass_illustration.py
python visualize_point_mass_results.py
python visualize_sgr_results.py
python visualize_sgr_her_results.py
python visualize_high_dim_point_mass_results.py