from DLpre import *
from L4_orthoDL import init_A, L4dl, get_power_after_transmission, inspect_W
import numpy as np

# influence of att magnitude and condition num
dimensions = 6
att_db = torch.tensor(np.linspace(-8, -8 - 10, dimensions), dtype=torch.complex128)
att = 10 ** (att_db * 0.1)
num_iter = 20
p_x_peak = 1
p_x_var = peakreal_2_var(p_x_peak)
H1 = gen_ch(dimensions, att)
u, s, v = H1.svd()
p_after_H = get_power_after_transmission(p_x_var, att)


##### 1. test by only L4ortho algorithm
def dl_WZF(att, u, v):
    '''
    run one dictionary learning and decompose a ZF equalizer of a specific channel
    :param att:channel att
    :param u: channel u
    :param v: channel v
    :return: decomposed sparse filter W, spasity list s_lst
    '''
    H = u.mm(torch.diag(att)).mm(v.conj().T)
    WZF_fil = WZF(H)
    A, W, s_lst = L4dl(WZF_fil, num_iter)
    return W, s_lst


att0_list = [0, -2, -4, -6, -8]
cond_list = [0, 2, 5, 7, 9, 11]
# 1a. influence of value of channel att, where condition numbers are fixed
# cond = cond_list[-1]
# W_lst = []
# s_lst_lst = []
# for att0 in att0_list:
#     att_db = torch.tensor(np.linspace(att0, att0 - cond, dimensions), dtype=torch.complex128)
#     att = att_db_to_att(att_db)
#     W, s_lst = dl_WZF(att, u, v)
#     W_lst.append(W)
#     s_lst_lst.append(s_lst)
#
# fig0, axs0 = plt.subplots(len(att0_list), 1, sharex=True)
# fig1, axs1 = plt.subplots(len(att0_list), 1)
# for i, W, s_lst in zip(list(range(len(W_lst))), W_lst, s_lst_lst):
#     lll = W.abs().reshape((-1,))
#     num_bins = 10 * ((lll.max() - lll.min()) / 1.).ceil()
#     axs0[i].hist(lll.numpy(), bins=num_bins.int(), rwidth=0.8)
#     axs1[i].plot(s_lst)


# 1b. influence of condition numbers, where value of channel att are fixed
att0 = att0_list[2]
W_lst = []
s_lst_lst = []
for cond in cond_list:
    att_db = torch.tensor(np.linspace(att0, att0 - cond, dimensions), dtype=torch.complex128)
    att = att_db_to_att(att_db)
    W, s_lst = dl_WZF(att, u, v)
    W_lst.append(W)
    s_lst_lst.append(s_lst)

fig0, axs0 = plt.subplots(len(cond_list), 1, sharex=True)
fig1, axs1 = plt.subplots(len(cond_list), 1)
for i, W, s_lst in zip(list(range(len(W_lst))), W_lst, s_lst_lst):
    lll = W.abs().reshape((-1,))
    num_bins = 10 * ((lll.max() - lll.min()) / 1.).ceil()
    axs0[i].hist(lll.numpy(), bins=num_bins.int(), rwidth=0.8)
    axs1[i].plot(s_lst)