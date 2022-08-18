import torch
import numpy as np
import matplotlib.pyplot as plt
from DLpre import gen_skew_hermitian, gen_unitary, gen_ch, WMMSE, peakreal_2_var, gen_QAM_X, prop, equalize, BERcounter, \
    plot_constellation, WZF


def init_A(dimensions):
    return gen_unitary(gen_skew_hermitian(dimensions))


def L4dl(Y, T):
    '''

    :param Y: data matrix
    :param T:num of iterations
    :return: (updated spasifying matrix A, sparse representation X=A.T Y, sparsity of X: X.norm(4)**4/np)

    '''
    n = Y.shape[0]
    p = Y.shape[1]
    A = init_A(n)
    s_lst = []  # list of spasities
    for t in range(T):
        ay = A.mm(Y)
        dA = 4 * ay.mul(ay).mul(ay).mm(Y.T.conj())
        u, s, v = torch.svd(dA)
        A = u.mm(v.conj().T)
        X = A.mm(Y)
        sparsity = X.reshape((-1,)).norm(4) ** 4 / (3. * n * p)
        s_lst.append(sparsity)

    return A.T.conj(), X, s_lst


def get_power_after_transmission(p_x_var, att):
    """
    calculate the signal power after transmitted by MIMO channels
    :param p_x_var: original signal power of one channel
    :param att: attenuation vec of MIMO channels
    :return: sig power after transmission
    """
    return torch.sqrt(p_x_var) * att.norm() ** 2


def inspect_W(W, s_lst):
    """
    to look at the sparsity condition of sparse W
    :param W: sparse digital equalizer
    :return: plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(s_lst)
    ax2.hist(W.abs().reshape((-1,)), bins=5)


def clip_W(W, ktop):
    """
    clip W with k largest elements remained
    :param W: W waiting to be clipped to be sparse
    :param ktop: k largest elements remained
    :return: clipped W
    """
    abw = W.abs()
    th = abw.view(W.numel()).topk(ktop)[0][-1]
    return W * ~(abw < th)





def WMMSE_BER_eval(H, SNR_after_H, ktop):
    # p_x_var = peakreal_2_var(p_x_peak)
    var_noise = p_after_H / SNR_after_H / H.shape[0]
    # propagation and load noise
    r_c = prop(X, H, var_noise)

    ##WMMSE way and count BER
    Wmmse = WMMSE(H, p_x_peak, var_noise)
    r_MMSE = equalize(Wmmse, r_c)
    BER_MMSE = BERcounter(X, r_MMSE)

    ##OEM way and count BER
    A, W, s_lst = L4dl(Wmmse, num_iter)
    # inspect_W(W, s_lst)
    # ktop = 10  # remain largest ktop values
    W = clip_W(W, ktop)
    r_OEM = equalize(A.mm(W), r_c)
    BER_OEM = BERcounter(X, r_OEM)

    # ##ZF way and count BER
    # Wzf = WZF(H)
    # r_ZF = equalize(Wzf, r_c)
    # BER_ZF = BERcounter(X, r_ZF)

    # ##OEZ way and count BER
    # Az, Wz, sz_lst = L4dl(Wzf, num_iter)
    # inspect_W(Wz, sz_lst)
    # # ktop = 10  # remain largest ktop values
    # Wz = clip_W(Wz, ktop)
    # r_OEZ = equalize(Az.mm(Wz), r_c)
    # BER_OEZ = BERcounter(X, r_OEZ)

    return BER_MMSE, BER_OEM
        # , BER_ZF, BER_OEZ




