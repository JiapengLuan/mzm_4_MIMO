from DLpre import gen_skew_symmetric,Giv
import torch
import numpy as np
import matplotlib.pyplot as plt

Ush=gen_skew_symmetric(4)
Aexp=torch.matrix_exp(Ush)
AGiv=Giv(Ush)
print('Aexp=',Aexp)
print('AGiv=',AGiv)
