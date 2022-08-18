import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns


def if_unitary(input_matrix):
    """
    if input_matrix is unitary
    :param input_matrix: tensor
    :return: True or False, bool
    """
    return torch.isclose(input_matrix.det().abs(), torch.tensor(1.))


def db_to_times(input_tensor):
    return 10 ** (input_tensor * 0.1)

def times_to_db(input_tensor):
    return 10 * input_tensor.log10()


def gen_skew_hermitian(dimensions):
    mat_rand = torch.randn(dimensions, dimensions, dtype=torch.complex64)
    return mat_rand - mat_rand.T.conj()


def gen_skew_symmetric(dimensions):
    mat_rand = torch.randn(dimensions, dimensions)
    return mat_rand - mat_rand.T

def get_max_id(input_mat):
    max_val=input_mat.max()

    for i in range(input_mat.shape[0]):
        if torch.isclose(input_mat[i],max_val):
            return i

def polar_decompose(input_mat):
    u, s, vh = torch.linalg.svd(input_mat)
    return vh.conj().T.mm(u.conj().T)

def get_random_unitary(dimensions):
    Ush = gen_skew_hermitian(dimensions)
    return torch.matrix_exp(Ush)

def sh_to_unitary(Ush):
    return torch.matrix_exp(Ush)

def gen_ch(dimensions, att):
    datt = torch.diag(att)
    Ush, Vsh = gen_skew_hermitian(dimensions), gen_skew_hermitian(dimensions)
    U, V = sh_to_unitary(Ush), sh_to_unitary(Vsh)
    H = V.conj().T.mm(datt).mm(U)
    return H


def wrap_phase(input_phase, phase_range=[-1.0 * np.pi, 1.0 * np.pi]):
    """
    Wrap phase between a range. default in range [-np.pi, np.pi]
    :param input_phase:
    :param phase_range:
    :return: warpped phase
    """
    period = phase_range[1] - phase_range[0]
    res = (input_phase - phase_range[0]) % period + phase_range[0]
    return res


def analyze_matrix(input_matrix):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    amplitude_info = input_matrix.abs()
    phase_info = input_matrix.angle()
    phase_info_wrap = wrap_phase(phase_info)
    sns.heatmap(amplitude_info, annot=True, fmt="d", linewidths=.5, ax=ax1, cbar=True).set(title='Amplitude')
    sns.heatmap(phase_info_wrap, annot=True, fmt="d", linewidths=.5, ax=ax2, cbar=True, cmap="mako", center=0.0).set(
        title='phase')
    return {'Amplitude': amplitude_info, 'phase': phase_info_wrap}

def insert_givens(givens,dimensions,i,j):
    res=torch.eye(dimensions,dtype=torch.complex64)
    res[i,i]=givens[0,0]
    res[i,j]=givens[0,1]
    res[j,i]=givens[1,0]
    res[j,j]=givens[1,1]
    return res

# print('test:',sys.argv)
