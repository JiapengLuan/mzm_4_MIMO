import torch
import numpy as np
import matplotlib.pyplot as plt
from helper import *
from devices import *
from MZImesh import Reck

dimension = 4
U = get_random_unitary(dimension)
Reck_base = Reck(dimension)
Reck_phase = Reck_base.matrix_to_phase(U)

Test_matrix = Reck_base.phase_to_matrix(Reck_phase)

result_test = torch.isclose(Test_matrix, np.exp(1.0j * Reck_base.common_phase) * U)
