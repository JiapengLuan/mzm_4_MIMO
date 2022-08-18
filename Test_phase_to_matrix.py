import torch
import numpy as np
import matplotlib.pyplot as plt
from helper import *
from devices import *
from MZImesh import Reck

dimension=4

Reck_base = Reck(dimension)

Reck_phase={'row_1': [torch.tensor(-1.0653),
                      torch.tensor([[ 0.0000, -0.6175]])],
 'row_2': [torch.tensor(-4.0113),
           torch.tensor([[ 2.9173, -0.2688],
                         [ 0.0000, -0.6757]])],
 'row_3': [torch.tensor(0.2707),
           torch.tensor([[-3.2104, -0.8464],
                         [-3.9498, -0.7306],
                         [ 0.0000, -0.7644]])],
 'row_4': torch.tensor([-3.5920, -3.5311, -3.4489])}

Test_matrix = Reck_base.phase_to_matrix(Reck_phase)

