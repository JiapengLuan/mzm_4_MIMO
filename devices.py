import torch
import numpy as np
import matplotlib.pyplot as plt
from helper import *


class mzi_unit():
    def __init__(self):
        pass

    def _directional_coupler(self):
        dc = 1.0 / np.sqrt(2.0) * torch.tensor([[1.0, 1.0j], [1.0j, 1.0]], dtype=torch.complex64)
        return dc

    def _phase_shifter_symmetric(self, phi_up, phi_down):
        ps = torch.tensor([[np.exp(1.0j * phi_up), 0.0], [0.0, np.exp(1.0j * phi_down)]], dtype=torch.complex64)
        return ps

    def _get_mzi_sym(self, theta, phi):
        '''
        get symmetric mzi.
        phi_up=theta+phi
        phi_down=theta-phi
        '''
        dc1 = self._directional_coupler()
        dc2 = self._directional_coupler()
        ps = self._phase_shifter_symmetric(phi_up=theta + phi, phi_down=theta - phi)
        return dc1.mm(ps).mm(dc2)

    def _get_mzi_asym(self):
        raise NotImplementedError("asymmetric mzi unit has not been written")

    def __call__(self, theta, phi):
        return self._get_mzi_sym(theta, phi)


class phase_shifter():
    def __init__(self):
        pass

    def _get_single_phase_shifter(self, phi):
        return torch.tensor([[np.exp(1.0j * phi)]])

    def __call__(self, phi):
        return self._get_single_phase_shifter(phi)


class givens_matrix():
    def __init__(self, dimension):
        self.dimension = dimension

    def _get_givens_matrix(self, index, theta, phi):
        assert index[1] - 1 == index[0]
        mzi = mzi_unit()(theta, phi)
        n_up = index[0]
        n_down = self.dimension - index[1] - 1
        Iup = torch.eye(n_up, n_up, dtype=mzi.dtype)
        Idown = torch.eye(n_down, n_down, dtype=mzi.dtype)
        A = torch.zeros(n_up, n_down + 2, dtype=mzi.dtype)
        B = torch.zeros(2, n_up, dtype=mzi.dtype)
        C = torch.zeros(2, n_down, dtype=mzi.dtype)
        D = torch.zeros(n_down, n_up + 2, dtype=mzi.dtype)

        block1 = torch.cat((Iup, A), 1)
        block2 = torch.cat((B, mzi, C), 1)
        block3 = torch.cat((D, Idown), 1)

        whole_givens = torch.cat((block1, block2, block3), 0)
        return whole_givens

    def __call__(self, index, phases_mzi):
        theta, phi = phases_mzi[0], phases_mzi[1]
        return self._get_givens_matrix(index, theta, phi)


class P_matrix():
    def __init__(self, dimension):
        self.dimension = dimension

    def _get_P_matrix(self, index, phase_list):
        I = torch.eye(self.dimension, dtype=torch.complex64)
        for id, phase in zip(index, phase_list):
            I[id, id] = np.exp(1.0j * phase)

        return I

    def __call__(self, index, phase_list):
        return self._get_P_matrix(index, phase_list)


class phGivens():
    def __init__(self, dimension):
        self.dimension = dimension

    def _single_givens(self, phase_list):
        theta = phase_list[0]  # mid phase diff
        phi = phase_list[1]  # mid phase common
        alpha1 = phase_list[2]  # left bottom phase
        alpha2 = phase_list[3]  # right bottom phase
        pG = torch.exp(1j * (phi + np.pi / 2)) * torch.tensor(
            [[torch.sin(theta), torch.exp(1j * alpha1) * torch.cos(theta)],
             [torch.cos(theta) * torch.exp(1j * alpha2), -torch.exp(1j * (alpha1 + alpha2)) * torch.sin(theta)]],
            dtype=torch.complex64)
        return pG

    def _map_mat_to_pG(self,mat):
        """mat: 2*2 unitary mat"""
        assert if_unitary(mat)
        a11=mat[0,0]
        a12=mat[0,1]
        a21=mat[1,0]
        a22=mat[1,1]
        phi=a11.angle()-np.pi/2.
        theta=torch.atan(a11.abs()/a12.abs())
        alpha1=a12.angle()-phi-np.pi/2.
        alpha2=a21.angle()-phi-np.pi/2.
        phase_list=wrap_phase(torch.tensor([theta,phi,alpha1,alpha2]))
        assert torch.isclose(self._single_givens(phase_list),mat)
        return phase_list

    def phase_to_pg(self, phase_list, index):
        return insert_givens(self._single_givens(phase_list), self.dimension, index[0], index[1])
