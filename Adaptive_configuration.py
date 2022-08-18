import torch
import numpy as np
import matplotlib.pyplot as plt
from helper import *
from channel import Channels


class CD_eq():
    def __init__(self, dimension=4):
        self.dimension = dimension
        # self.init_G_list = self._random_init()
        self.init_G_list = self._Identity_init()

    def _Identity_init(self):
        """
        all mzis are initialized by I matrices
        :return list of pG matrices [g1,g2,...]
        """
        d = self.dimension
        self.index_list = [[2, 3], [1, 2], [2, 3], [0, 1], [1, 2], [2, 3]]
        imats = [torch.eye(2, dtype=torch.complex64)] * int(d * (d - 1) / 2)
        inserted_imats = []
        for imat, idx in zip(imats, self.index_list):
            inserted_imats.append(insert_givens(imat, self.dimension, idx[0], idx[1]))
        return inserted_imats

    def _random_init(self):
        d = self.dimension
        self.index_list = [[2, 3], [1, 2], [2, 3], [0, 1], [1, 2], [2, 3]]
        imats = [get_random_unitary(2) for _ in range(int(d * (d - 1) / 2))]
        inserted_imats = []
        for imat, idx in zip(imats, self.index_list):
            inserted_imats.append(insert_givens(imat, self.dimension, idx[0], idx[1]))
        return inserted_imats

    def _cov_init(self, att_db):
        H = Channels(self.dimension).random_channel(att_db=att_db)
        return H.conj().T

    def _CD_polar(self, cov, update_method='cyclic'):
        G_list = self.init_G_list[:]
        Wh = torch.linalg.multi_dot(G_list[::-1])
        W = Wh.conj().T
        Zh = cov.mm(W)
        Z = Zh.conj().T

        trace_loss_init = self._evaluate_trace(cov, Zh)
        trace_loss_list = [trace_loss_init]
        mat_error_loss_init = self._evaluate_mat_error(cov, W)
        mat_error_list = [mat_error_loss_init]
        kf_norms = []
        gd_norms = []
        Wh_list = []
        gh_prime_list = []
        for i in range(1000):


            G_list, gh_prime = self._update_single_mzi(mzi_id, G_list, Zh)

            Wh = torch.linalg.multi_dot(G_list[::-1])
            W = Wh.conj().T
            Zh = cov.mm(W)

            trace_loss = self._evaluate_trace(cov, Zh)
            trace_loss_list.append(trace_loss)
            mat_error_loss = self._evaluate_mat_error(cov, W)
            mat_error_list.append(mat_error_loss)
            kf_norms.append(kf_norm_list)
            gd_norms.append(gd_norm_list)
            Wh_list.append(Wh)
            gh_prime_list.append(gh_prime)
        return trace_loss_list, mat_error_list, kf_norms, gd_norms, Wh_list, gh_prime_list

    def _choose_index_mzi_cyclic(self, i):
        return 5 - i % 6

    def _choose_index_mzi_max(self, Z, G_list):
        Zh = Z.conj().T
        pos_list = self.index_list
        kf_norm_loss_list = []
        for gid in range(len(G_list) - 1, -1, -1):
            this_id = pos_list[gid]
            this_sub_mat = Zh[this_id[0]:this_id[1] + 1, this_id[0]:this_id[1] + 1]
            this_kf_norm = torch.linalg.norm(this_sub_mat, ord='nuc')
            this_kf_norm_loss = this_kf_norm - torch.trace(this_sub_mat).real
            assert this_kf_norm_loss >= -1e-4
            kf_norm_loss_list.insert(0, this_kf_norm_loss)
            Zh = G_list[gid].conj().T.mm(Zh).mm(G_list[gid])
        kf_norm_loss_list = torch.tensor(kf_norm_loss_list)
        max_val = kf_norm_loss_list.max()
        max_id = get_max_id(kf_norm_loss_list)
        return max_id, max_val, kf_norm_loss_list

    def _choose_index_mzi_Rmax(self, Z, G_list):
        Zh = Z.conj().T
        pos_list = self.index_list
        rmgrad_list = []  # riemanian gradient grad(I2)
        for gid in range(len(G_list) - 1, -1, -1):
            this_id = pos_list[gid]
            rmgrad_list.insert(0, this_grad)
            Zh = G_list[gid].conj().T.mm(Zh).mm(G_list[gid])
        rmgrad_list = torch.tensor(rmgrad_list)
        max_val = rmgrad_list.max()
        max_id = get_max_id(rmgrad_list)
        return max_id, max_val, rmgrad_list

    def _update_single_mzi(self, update_id, G_list, Zh):
        num_mzi_flip = len(G_list) - (update_id + 1)  # num of mzi need flipping
        pos_list = self.index_list
        pos_upd = pos_list[update_id]
        for i in range(num_mzi_flip):
            mzi_to_flip = G_list[-(i + 1)]
            Zh = mzi_to_flip.conj().T.mm(Zh).mm(mzi_to_flip)
        sub_Zh = Zh[pos_upd[0]:pos_upd[1] + 1, pos_upd[0]:pos_upd[1] + 1]
        u, s, vh = torch.linalg.svd(sub_Zh)
        gh_prime_not_insert = vh.conj().T.mm(u.conj().T)
        gh_prime = insert_givens(gh_prime_not_insert, self.dimension, pos_upd[0], pos_upd[1])
        G_list[update_id] = gh_prime.conj().T.mm(G_list[update_id])
        return G_list, gh_prime_not_insert

    def _evaluate_trace(self, cov, Zh):
        trace_to_lower_bound = torch.linalg.norm(cov, ord='nuc') - torch.trace(Zh).real
        return trace_to_lower_bound

    def _evaluate_mat_error(self, cov, W):
        vhu = polar_decompose(cov)
        return torch.linalg.norm(vhu - W, ord='fro') / torch.linalg.norm(vhu, ord='fro')





