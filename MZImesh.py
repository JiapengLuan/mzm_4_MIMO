import torch
import numpy as np
import matplotlib.pyplot as plt
from helper import *
from devices import *


class Reck():
    def __init__(self, dimension):
        '''
        n:num of input output ports. n*n matrix
        '''
        self.dimension = dimension
        self.common_phase = None

    def _get_Reck(self):
        '''get unitary matrix in Reck form
        '''

    def _one_row_in_Reck(self, j, phase_ps, phase_mzi_list):
        """
        j: j-th row of Reck
        phase_ps:phase of input phase shifter
        phase_mzi_list:phases of mzis of this row. [[theta_mzi_j,phi_mzi_j],[theta_mzi_j-1,phi_mzi_j-1],...]. Order: right to left phase.
        """
        assert len(phase_mzi_list) == j
        P = P_matrix(self.dimension)([j], [phase_ps])
        mzi_index_list = [[index, index + 1] for index in range(0, j)]
        givens_base = givens_matrix(self.dimension)
        givens_matrix_list = []
        for k in range(len(mzi_index_list)):
            givens_matrix_list.append(givens_base(mzi_index_list[k], phase_mzi_list[k]))

        whole_mat_list = givens_matrix_list + [P]

        return torch.linalg.multi_dot(whole_mat_list)

    def _Q_row(self, phase_list_Q):
        """
        last row of Reck, composed by an array of phase shifters
        :param phase_list_Q: list. phases of last row of Reck. Order: right to left phase.
        :return: transfer matrix of Q row
        """
        assert len(phase_list_Q) == self.dimension - 1
        P_base = P_matrix(self.dimension)
        P_list = [P_base([i + 1], [phase_ps]) for i, phase_ps in enumerate(phase_list_Q)]
        if not len(P_list) == 1:
            res = torch.linalg.multi_dot(P_list)
        else:
            res = P_list[0]
        return res

    def phase_to_matrix(self, Reck_phase_dict):
        """
        given phase setting of all phase shifters, get transfer matrix
        :param Reck_phase_dict: {"row_1": phase_row_1,...,"row_m": phase_row_m}. For j = 1,...,m-1, phase_row_j=[input_ps_phase,[[theta_mzi_1,phi_mzi_1],[theta_mzi_2,phi_mzi_2],...]]. For j = m, phase_row_j = [output_ps_1,...,output_ps_m-1]
        :return: transfer matrix of the whole mzi mesh
        """
        num_input_rows = len(Reck_phase_dict)
        assert num_input_rows == self.dimension
        row_matrix_list = []
        for row_idx in range(1, num_input_rows):
            this_row_phases = Reck_phase_dict[f'row_{row_idx}']
            this_row_input_ps_phase = this_row_phases[0]
            this_row_mzi_phases = this_row_phases[1]
            # this_row_mzi_phases.reverse()
            this_row_matrix = self._one_row_in_Reck(row_idx, this_row_input_ps_phase, this_row_mzi_phases.flip([0]))
            row_matrix_list.append(this_row_matrix)
        # Reck_phase_dict[f'row_{num_input_rows}'].reverse()
        Q_phase_list = Reck_phase_dict[f'row_{num_input_rows}']
        Q_row_matrix = self._Q_row(Q_phase_list.flip([0]))
        row_matrix_list.append(Q_row_matrix)
        row_matrix_list.reverse()
        return torch.linalg.multi_dot(row_matrix_list)

    def _init_Reck_phase_dict(self):
        Reck_dict = {f'row_{j}': [torch.zeros(1), torch.zeros(j, 2)] for j in range(1, self.dimension)}
        Reck_dict[f'row_{self.dimension}'] = torch.zeros(self.dimension - 1)
        return Reck_dict

    def matrix_to_phase(self, input_matrix):
        # if unitary
        m = self.dimension
        assert if_unitary(input_matrix)
        # init result phases
        Reck_dict = self._init_Reck_phase_dict()

        aux_matrix = input_matrix.conj()

        # j: j-th row. k:k-th mzi.
        for j in range(1, m):
            assert f'row_{j}' in Reck_dict
            this_row_input_ps_phase = aux_matrix[m - 1, j - 1].angle() - aux_matrix[m - 1, j].angle()
            Reck_dict[f'row_{j}'][0] = this_row_input_ps_phase
            aux_matrix = aux_matrix.mm(P_matrix(m)(index=[j], phase_list=[this_row_input_ps_phase]))
            for k in range(j):
                Vleft = aux_matrix[m - 1 - k, j - 1 - k]
                Vright = aux_matrix[m - 1 - k, j - 1 - k + 1]
                # assert torch.isclose(Vleft.angle(),Vright.angle())
                this_mzi_phi = np.arctan(-Vright.abs() / Vleft.abs())
                Reck_dict[f'row_{j}'][1][k, 1] = this_mzi_phi
                if not k == j - 1:
                    V_upleft = aux_matrix[m - 2 - k, j - 2 - k]
                    V_upmid = aux_matrix[m - 2 - k, j - 1 - k]
                    V_upright = aux_matrix[m - 2 - k, j - 0 - k]
                    this_mzi_theta = V_upleft.angle() - (
                                V_upmid * np.sin(this_mzi_phi) + V_upright * np.cos(this_mzi_phi)).angle() - np.pi / 2.0
                    Reck_dict[f'row_{j}'][1][k, 0] = this_mzi_theta
                this_givens = givens_matrix(m)(index=[j - k - 1, j - k], phases_mzi=[Reck_dict[f'row_{j}'][1][k, 0],
                                                                                     Reck_dict[f'row_{j}'][1][k, 1]])
                aux_matrix = aux_matrix.mm(this_givens)

        for k in range(1, m):
            this_q_phase = aux_matrix[0, 0].angle() - aux_matrix[m - k, m - k].angle()
            Reck_dict[f'row_{m}'][k - 1] = this_q_phase
            aux_matrix = aux_matrix.mm(P_matrix(m)(index=[m - k], phase_list=[this_q_phase]))

        self.common_phase = aux_matrix[0, 0].angle()

        return Reck_dict
