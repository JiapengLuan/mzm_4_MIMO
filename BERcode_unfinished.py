

####### mdl big to small
SNRlist=list(range(3,15,1))
seql=[5e6,1e7,1.5e7]
ktopl=[27,27,27]
mdll=[7,6,5]


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_yscale('log')
ax.set_ylim([1e-5, 1e0])
ax.set_xlim([3, 14])
ax.set_xlabel('OSNR(dB)',fontsize=18)
ax.set_ylabel('Averaged BER',fontsize=18)
ax.grid()

# generate channel
dimensions = 6
# att = torch.tensor([1, .8, .5, .3], dtype=torch.complex128)
# att_db = torch.tensor([-8, -10, -11, -12, -14, -16], dtype=torch.complex128)
att_db = torch.tensor(np.linspace(-8,-8-mdll[0],dimensions), dtype=torch.complex128)
att = 10 ** (att_db * 0.1)
num_iter = 20

p_x_peak = 1
p_x_var = peakreal_2_var(p_x_peak)
# var_noise = 0.03
H1 = gen_ch(dimensions, att)
u,s,v=H1.svd()
p_after_H = get_power_after_transmission(p_x_var, att)
# A, W, s_lst = L4dl(H, num_iter)

# generate ground truth signal
sequence_length = int(seql[0])
X = gen_QAM_X(dimensions, sequence_length, 4, p_x_peak)

# SNR_after_H_db_list = torch.tensor(list(range(5, 18, 2)))
SNR_after_H_db_list = torch.tensor(SNRlist)
ktop = ktopl[0]  # remain largest ktop values
SNR_after_H_list = (10 ** (SNR_after_H_db_list * 0.1)).to(torch.complex128)
BER_MMSE_list1 = []
BER_OEM_list1 = []
# BER_ZF_list = []
# BER_OEZ_list = []
for idx in range(SNR_after_H_list.shape[0]):
    BM, BEM = WMMSE_BER_eval(H1, SNR_after_H_list[idx], ktop)
    BER_MMSE_list1.append(BM)
    BER_OEM_list1.append(BEM)
    # BER_ZF_list.append(BZ)
    # BER_OEZ_list.append(BEZ)
    print(f'{idx}-th finished')

ax.plot(SNRlist,BER_MMSE_list1, color='red',label='MDL=14dB, MMSE')
ax.plot(SNRlist,BER_OEM_list1, color='red', linestyle='dashed',label='MDL=14dB, OE')
# line11.set_label()
####################################


# generate channel
dimensions = 6
# att = torch.tensor([1, .8, .5, .3], dtype=torch.complex128)
# att_db = torch.tensor([-8, -10, -11, -12, -14, -16], dtype=torch.complex128)
att_db = torch.tensor(np.linspace(-8,-8-mdll[1],dimensions), dtype=torch.complex128)
att = 10 ** (att_db * 0.1)
num_iter = 20

p_x_peak = 1
p_x_var = peakreal_2_var(p_x_peak)
# var_noise = 0.03
H2 = u.mm(torch.diag(att)).mm(v.conj().T)
p_after_H = get_power_after_transmission(p_x_var, att)
# A, W, s_lst = L4dl(H, num_iter)

# generate ground truth signal
sequence_length = int(seql[1])
X = gen_QAM_X(dimensions, sequence_length, 4, p_x_peak)

# SNR_after_H_db_list = torch.tensor(list(range(5, 18, 2)))
SNR_after_H_db_list = torch.tensor(SNRlist)
ktop = ktopl[1]  # remain largest ktop values
SNR_after_H_list = (10 ** (SNR_after_H_db_list * 0.1)).to(torch.complex128)
BER_MMSE_list2 = []
BER_OEM_list2 = []
# BER_ZF_list = []
# BER_OEZ_list = []
for idx in range(SNR_after_H_list.shape[0]):
    BM, BEM = WMMSE_BER_eval(H2, SNR_after_H_list[idx], ktop)
    BER_MMSE_list2.append(BM)
    BER_OEM_list2.append(BEM)
    # BER_ZF_list.append(BZ)
    # BER_OEZ_list.append(BEZ)
    print(f'{idx}-th finished')

ax.plot(SNRlist,BER_MMSE_list2, color='blue',label='MDL=12dB, MMSE')
ax.plot(SNRlist,BER_OEM_list2, color='blue', linestyle='dashed',label='MDL=12dB, OE')
#########################################################

# generate channel
dimensions = 6
# att = torch.tensor([1, .8, .5, .3], dtype=torch.complex128)
# att_db = torch.tensor([-8, -10, -11, -12, -14, -16], dtype=torch.complex128)
att_db = torch.tensor(np.linspace(-8,-8-mdll[2],dimensions), dtype=torch.complex128)
att = 10 ** (att_db * 0.1)
num_iter = 20

p_x_peak = 1
p_x_var = peakreal_2_var(p_x_peak)
# var_noise = 0.03
H3 = u.mm(torch.diag(att)).mm(v.conj().T)
p_after_H = get_power_after_transmission(p_x_var, att)
# A, W, s_lst = L4dl(H, num_iter)

# generate ground truth signal
sequence_length = int(seql[2])
X = gen_QAM_X(dimensions, sequence_length, 4, p_x_peak)

# SNR_after_H_db_list = torch.tensor(list(range(5, 18, 2)))
SNR_after_H_db_list = torch.tensor(SNRlist)
ktop = ktopl[2]  # remain largest ktop values
SNR_after_H_list = (10 ** (SNR_after_H_db_list * 0.1)).to(torch.complex128)
BER_MMSE_list3 = []
BER_OEM_list3 = []
# BER_ZF_list = []
# BER_OEZ_list = []
for idx in range(SNR_after_H_list.shape[0]):
    BM, BEM = WMMSE_BER_eval(H3, SNR_after_H_list[idx], ktop)
    BER_MMSE_list3.append(BM)
    BER_OEM_list3.append(BEM)
    # BER_ZF_list.append(BZ)
    # BER_OEZ_list.append(BEZ)
    print(f'{idx}-th finished')

ax.plot(SNRlist,BER_MMSE_list3, color='green',label='MDL=10dB, MMSE')
ax.plot(SNRlist,BER_OEM_list3, color='green', linestyle='dashed',label='MDL=10dB, OE')

ax.legend()



# plt.plot(BER_MMSE_list)
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(BER_MMSE_list, color='red')
# ax.plot(BER_OEM_list, color='red', linestyle='dashed')
# ax.plot(BER_ZF_list, color='green')
# ax.plot(BER_OEZ_list, color='green', linestyle='dashed')
# ax.set_yscale('log')
# ax.set_ylim([1e-5, 1e0])

# fig,(ax1,ax2)=plt.subplots(2,1)
# ax1.plot(s_lst)
# ax2.hist(X.abs().reshape((-1,)),bins=5)
# q,r=torch.qr(Y)
# s_r=r.reshape((-1,)).norm(4) ** 4 / (3. * r.shape[0] * r.shape[1])