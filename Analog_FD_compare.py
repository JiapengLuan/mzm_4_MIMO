import torch
import numpy as np
import matplotlib.pyplot as plt
from channel import Channels
from helper import *
from DLpre import gen_QAM_X, prop, equalize, BERcounter


class metric():
    def __init__(self, equilizer_mat, channel_mat, sigma_n):
        self.equalizer_mat = equilizer_mat
        self.channel_mat = channel_mat
        self.dimension = channel_mat.shape[0]
        self.sigma_n = sigma_n

    def _MSE_matrix(self):
        AA = self.equalizer_mat.T.conj().mm(self.channel_mat) - torch.eye(self.dimension, dtype=torch.complex64)
        noise_covariance = self.sigma_n.abs() ** 2 * torch.eye(self.dimension, dtype=torch.complex64)
        MSE_mat = AA.mm(AA.T.conj()) + self.equalizer_mat.T.conj().mm(noise_covariance).mm(self.equalizer_mat)
        return MSE_mat

    def loss_MSE(self):
        return torch.trace(self._MSE_matrix()).real

    def _transmission_power_matrix_with_equalization(self):
        WH=self.equalizer_mat.T.conj().mm(self.channel_mat)
        noise_covariance = self.sigma_n.abs() ** 2 * torch.eye(self.dimension, dtype=torch.complex64)
        nosie_vec_for_1_column=torch.diag(self.equalizer_mat.T.conj().mm(noise_covariance).mm(self.equalizer_mat))
        return WH.abs()**2+torch.column_stack([nosie_vec_for_1_column]*self.dimension)

    def _plot_3d_mat(self,input_mat,ax,title):
        _x = np.arange(self.dimension)+1
        _y = np.arange(self.dimension)+1
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()
        bottom=np.zeros_like(input_mat.reshape((-1,)))
        width = depth = 0.9
        top = input_mat.reshape((-1,)).abs()
        ax.bar3d(x, y, bottom, width, depth, top, shade=True)
        ax.set_zlabel('Transmission')
        ax.set_title(title)
        ax.set_zlim(0.,1.)

    def plot_3d_mat(self,input_mats,title):
        # input_mats=[self._transmission_power_matrix_with_equalization()]
        # fig, axes = plt.subplots(1, len(input_mats))
        fig = plt.figure(figsize=(8, 3))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        # for i,mat in enumerate(input_mats):
            # self._plot_3d_mat(mat,axes[0][i])
        self._plot_3d_mat(input_mats[0],ax1,title)
        self._plot_3d_mat(input_mats[1],ax2,title)



class BER_counter():
    def __init__(self, channel_mat, data_frame_length,num_chuncks, sigma_s=torch.tensor(1., dtype=torch.complex64)):
        # self.equalizer_mat = equilizer_mat
        self.channel_mat = channel_mat
        self.dimension = channel_mat.shape[0]
        # self.sigma_n = sigma_n
        self.sigma_s = sigma_s
        self.length = data_frame_length
        # self.source = self._signal_source(M=4)
        self.num_chuncks = num_chuncks
    def _signal_source(self, M=4):
        p_peak = torch.sqrt(self.sigma_s.abs() ** 2 / 2.)
        return gen_QAM_X(self.dimension, int(self.length/self.num_chuncks), M, p_peak)

    def _ber_count_single_n(self, equalizer, sigma_n):
        all_BER_list=[]
        for idx in range(self.num_chuncks):
            source = self._signal_source()
            sig_after_channel = prop(X=source, H=self.channel_mat, var_noise=sigma_n.abs() ** 2)
            sig_after_equalizer = equalize(mat_eq=equalizer, sig_in=sig_after_channel)
            BER_list = [BERcounter(source[idx], sig_after_equalizer[idx])[1] for idx in range(self.dimension)]
            whole_BER = BERcounter(source, sig_after_equalizer)[1]
            BER_list.append(whole_BER)#error bit counts
            all_BER_list.append(BER_list)
        error_counts_list=torch.tensor(all_BER_list).sum(0)
        all_counts_list=torch.tensor([self.length]*self.dimension+[self.length*self.dimension])
        final_BER_list=(error_counts_list/all_counts_list).tolist()
        return final_BER_list

    def ber_count(self, equalizer_list, snr_list):
        sigma_n_list = torch.sqrt(torch.tensor(1.) / db_to_times(snr_list))
        BER_result = [[] for _ in range(self.dimension + 1)]
        snr_length=sigma_n_list.shape[0]
        for idx in range(sigma_n_list.shape[0]):
            sigma_n = sigma_n_list[idx]
            ber_this_noise = self._ber_count_single_n(equalizer_list[idx], sigma_n)
            for idx_dim in range(self.dimension + 1):
                BER_result[idx_dim].append(ber_this_noise[idx_dim])
            print(f'this equalizer {100.*(idx+1)/snr_length}% finished')
        # self.visualize_ber(snr_list, BER_result)
        return BER_result  # [ch1_ber,...,chn_ber,whole_ber]

    def visualize_ber(self, snr_list, BER_result, ax1, ax2, linestyle,name,marker):
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        label_list = [f'channel{i + 1}_'+name for i in range(len(BER_result) - 1)]
        label_list.append('All_channel_'+name)
        color_list = ['b', 'g', 'r', 'c', 'k']
        for idx in range(len(BER_result) - 1):
            ax1.plot(snr_list.tolist(), BER_result[idx], label=label_list[idx], c=color_list[idx], linestyle=linestyle,marker=marker)
        ax2.plot(snr_list.tolist(), BER_result[-1], label=label_list[-1], c=color_list[-1], linestyle=linestyle,marker=marker)
        ax1.legend()
        ax2.legend()

        ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax1.set_xlabel('SNR(dB)',fontsize=15)
        ax2.set_xlabel('SNR(dB)',fontsize=15)
        ax1.set_ylabel('BER',fontsize=15)
        ax2.set_ylabel('BER',fontsize=15)

    def ber_compare(self, snr_list, Wfd_list, Wph_list):
        BERfd = self.ber_count(Wfd_list, snr_list)
        BERph = self.ber_count(Wph_list, snr_list)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.visualize_ber(snr_list, BERfd, ax1, ax2, linestyle='-',name='FD',marker='*')
        self.visualize_ber(snr_list, BERph, ax1, ax2, linestyle='--',name='PH',marker='^')
        # fig.grid()
        ax1.grid(True)
        ax2.grid(True)
        return BERfd, BERph


class FD_equalizer():
    def __init__(self, channel_mat, sigma_n):
        self.channel_mat = channel_mat
        self.dimension = channel_mat.shape[0]
        self.sigma_n = sigma_n

    def MMSE(self):
        noise_covariance = self.sigma_n.abs() ** 2 * torch.eye(self.dimension, dtype=torch.complex64)
        AAA = self.channel_mat.mm(self.channel_mat.conj().T) + noise_covariance
        AAAinv = torch.linalg.inv(AAA)
        return AAAinv.mm(self.channel_mat)


class ph_equalizer():
    def __init__(self, channel_mat, sigma_n):
        self.channel_mat = channel_mat
        self.dimension = channel_mat.shape[0]
        self.sigma_n = sigma_n

    def kkt(self):
        channel_cov = self.channel_mat.mm(self.channel_mat.conj().T)
        uch, sch, vhch = torch.linalg.svd(self.channel_mat)
        sigma_n_2 = self.sigma_n.abs() ** 2
        noise_cov = sigma_n_2 * torch.eye(self.dimension, dtype=torch.complex64)
        miu_list = []  # Largrangian multiplier
        for idx in range(sch.shape[0]):
            if sch.abs()[idx] >= torch.tensor(1.):
                this_miu = torch.tensor(0.)
            else:
                if sigma_n_2 <= noise_threshold[idx]:
                    this_miu = noise_threshold[idx] - sigma_n_2
                    assert this_miu >= torch.tensor(0.)
                else:
                    this_miu = torch.tensor(0.)
            miu_list.append(this_miu)
        smiu = torch.diag(torch.stack(miu_list)).to(dtype=torch.complex64)
        miu = uch.mm(smiu).mm(uch.conj().T)
        in_bracket = channel_cov + noise_cov + miu
        in_bracket_inv = torch.linalg.inv(in_bracket)
        equalizer = in_bracket_inv.mm(self.channel_mat)
        return equalizer





