import torch
import numpy as np
import matplotlib.pyplot as plt


def gen_skew_hermitian(dimensions):
    mat_rand = torch.randn(dimensions, dimensions, dtype=torch.complex64)
    return mat_rand - mat_rand.T.conj()

def gen_skew_symmetric(dimensions):
    mat_rand = torch.randn(dimensions, dimensions)
    return mat_rand - mat_rand.T

def Giv(Ush):
    n=Ush.shape[0] #get n of an n*n square matrix
    giv_list=[]
    for i in range(n-1):
        for j in range(i+1,n):
            giv_mat=get_givens_rotation_mat(Ush[i,j],i,j,n)
            giv_list.append(giv_mat)
    if len(giv_list)==1:
        giv_list.append(torch.eye(n))
    MUL_givs=torch.linalg.multi_dot(giv_list)
    return MUL_givs

def get_givens_rotation_mat(theta,i,j,dimensions):
    assert i<j
    out=torch.eye(dimensions)
    out[i,i]=torch.cos(theta)
    out[j,j]=torch.cos(theta)
    out[i,j]=-torch.sin(theta)#+sin(theta) for i,j based on book optimization over matrix mnifold
    out[j,i]=torch.sin(theta)
    return out




def gen_unitary(Ush):
    return torch.matrix_exp(Ush)


def check_uni(U):
    return print('det', U.det().abs(), 'self_mul', torch.matmul(U, U.conj().T))


def get_mat_power(U):
    return U.abs() ** 2


def gen_ch(dimensions, att):
    datt = torch.diag(att)
    Ush, Vsh = gen_skew_hermitian(dimensions), gen_skew_hermitian(dimensions)
    U, V = gen_unitary(Ush), gen_unitary(Vsh)
    H = V.conj().T.mm(datt).mm(U)
    return H

def att_db_to_att(att_db):
    return 10 ** (att_db * 0.1)

def WMMSE(H, px_var, var_noise):
    """
    :param H: channel matrix
    :param px: variance of input signal x
    :param var_noise: variance of noise, sigma**2
    :return: WMMSE matrix, without conj transpose h !
    """
    Nt=H.shape[0]#num of channels
    gama=px_var*Nt/(Nt*var_noise)
    mat = H.T.conj().mm(H) + (1/gama) * torch.eye(H.shape[0], dtype=torch.complex128)
    matinv = torch.linalg.inv(mat)
    res = matinv.mm(H.conj().T)  # this is Wmmse**h
    return res.conj().T

def WZF(H):
    mat=H.conj().T.mm(H)
    matinv = torch.linalg.inv(mat)
    res=matinv.mm(H.conj().T)
    return res.conj().T


def prop(X, H, var_noise):
    E_noise=(torch.tensor(var_noise)/2).sqrt()
    noise_vec = E_noise * torch.randn(X.shape, dtype=X.dtype)
    r_c = H.mm(X) + noise_vec  # signal before PTC
    return r_c


def equalize(mat_eq, sig_in):
    """

    :param mat_eq: equalize matrix
    :param sig_in: input signal, could be signal or matrix
    :return: output signal
    """
    return mat_eq.conj().T.mm(sig_in)

def peakreal_2_var(p):
    """
    convert peak E field of real part to var of signal (power) for QAM4
    :param p: peak E field of real
    :return: var of signal
    """
    return torch.tensor(2*p**2,dtype=torch.complex128)

def gen_QAM_X(dimensions, length, M, p_peak):
    """

    :param dimensions: mode channel num
    :param length: sequence lenght
    :param M: M-QAM modulation
    :param p_peak: x signal peak power of real part
    :return: X, signals of all channels
    """

    low = 0
    high = (torch.sqrt(torch.tensor(M))).to(torch.int32) - 1
    randnum_real = torch.randint(low, high + 1, (dimensions, length))
    randnum_imag = torch.randint(low, high + 1, (dimensions, length))

    ##normalize
    X_re = 2 * p_peak / (high - low) * randnum_real - p_peak
    X_im = 2 * p_peak / (high - low) * randnum_imag - p_peak
    X = X_re + 1j * X_im
    return X.to(torch.complex64)


def BERcounter(X,Y):
    """
    Count bit error rate for QAM-4
    :param X: ground truth data
    :param Y: processed data
    :return: bit error rate
    """
    ang_diff=(X.angle()-Y.angle()).abs()
    mat_bool=~(ang_diff<=np.pi/4.)
    error_counts=mat_bool.count_nonzero()
    all_counts=X.numel()
    BER=1.*error_counts/all_counts
    return BER, error_counts, all_counts



def plot_constellation(X):
    """

    :param X: X is 1d complex signal
    :return: constellation diagram
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    s=1
    c='r'
    ax.scatter(X[0,0:5000].real, X[0,0:5000].imag,s,c)
    plt.show()


#######DL start here

def soft_th(eps, alpha):
    k_plus = eps.abs() - alpha if ((eps.abs() - alpha > 0)) else 0
    return torch.exp(1j * eps.angle()) * k_plus


def init_A(Wmmse):
    u, s, v = Wmmse.svd()
    return u


def get_iter_A(W, Wmmse):
    u, s, v = torch.svd(W.mm(Wmmse.conj().T))
    return v.mm(u.conj().T)


def get_iter_W(A, Wmmse, lam, W):
    E = Wmmse - A.mm(W)
    dim = A.shape[1]
    wjnew_lst = []
    for j in range(A.shape[1]):
        aj = A[:, j].view((dim, 1))
        wj = W[j, :].view((1, dim))
        F = E + aj.mm(wj)
        fl_lst = [F[:, l].view((1, dim)) for l in range(dim)]
        wjl_lst = [soft_th(fl.mm(aj), lam / 2) for fl in fl_lst]
        wj_new = torch.cat(wjl_lst, 1)
        W[j, :] = wj_new.view(W[j, :].shape)
        A = get_iter_A(W, Wmmse)
        wjnew_lst.append(wj_new)
        E = F - aj.mm(wj_new)
    if not wjnew_lst[0].shape[0] == 1:
        raise TypeError('wj dimension not correct, should be a row')
    Wnew = torch.cat(wjnew_lst, 0)
    return Wnew




# dimensions = 6
# length = 10000
# QAM_order = 4
# p_x_peak = 1
# var_noise = 0.03
# # var_noise=0
# att_db = torch.tensor([-8, -8.5, -9, -10, -10.5, -11], dtype=torch.complex128)
# att = 10 ** (att_db * 0.1)
# X = gen_QAM_X(dimensions, length, QAM_order, p_x_peak)
# H = gen_ch(dimensions, att)
# r_c = prop(X, H, var_noise)
# Wmmse = WMMSE(H, p_x_peak, var_noise)
# r = equalize(Wmmse, r_c)
#
# # plot_constellation(r)
#
# num_iter = 500
# lam = .1


##main DL
def run_dl(lam):
    norm_Wmmse=Wmmse.norm()
    A = init_A(Wmmse)
    W = A.conj().T.mm(Wmmse)  # init A and W
    loss_ori = []
    W1norm = []
    for i in range(num_iter):
        W = get_iter_W(A, Wmmse, lam, W)
        A = get_iter_A(W, Wmmse)
        loss_ori.append(((Wmmse - A.mm(W)).norm() ** 2)/(norm_Wmmse**2))
        W1norm.append(A.norm(p=1))
        if i % 20 == 0:
            print(f'progress:{i / num_iter * 100}%')
    plt.plot(loss_ori)
    # plt.plot(W1norm)
    Wtot = A.mm(W)
    r_dl = equalize(Wtot, r_c)
    # plot_constellation(r_dl)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # s = 0.5
    # c = 'r'
    # ax.scatter(r_dl[0].real, r_dl[0].imag, s, c)
    return W.abs()

# dimensions=4
# hhh=gen_skew_hermitian(dimensions)
# uni=gen_unitary(hhh)
# check_uni(uni)
