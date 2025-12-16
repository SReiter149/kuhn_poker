import numpy as np
import torch
from matplotlib import pyplot as plt
import pdb

from models import OptimalBotP2
from plot_research import plot_strategy_heatmap, plot_strategy_heatmap_p2




policy = OptimalBotP2()

plot_strategy_heatmap_p2(policy, device='cpu', save_path="results/optimal_p2_heatplot.png")
