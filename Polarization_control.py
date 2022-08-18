import torch
import numpy as np
import matplotlib.pyplot as plt


class Basis_change():
    def __init__(self):
        pass

    def _get_basis_change_matrix(self, ang):
        G = [[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]]
        A11 = np.cos(ang) * torch.tensor(G, dtype=torch.complex64)
        A12 = np.sin(ang) * torch.tensor(G, dtype=torch.complex64)
        A21 = -np.sin(ang) * torch.tensor(G, dtype=torch.complex64)
        A22 = np.cos(ang) * torch.tensor(G, dtype=torch.complex64)
        A1 = torch.cat((A11, A12), 1)
        A2 = torch.cat((A21, A22), 1)
        res = torch.cat((A1, A2), 0)
        return res

    def __call__(self, ang):
        return self._get_basis_change_matrix(ang)


class wave_plate():
    """optical field input: [Eax, Ebx, Eay, Eby]"""

    def __init__(self):
        pass

    def _diag(self, phi1, phi2, phi3, phi4):
        P = [[np.exp(1.0j * phi1), 0.0, 0.0, 0.0], [0.0, np.exp(1.0j * phi2), 0.0, 0.0],
             [0.0, 0.0, np.exp(1.0j * phi3), 0.0], [0.0, 0.0, 0.0, np.exp(1.0j * phi4)]]
        return torch.tensor(P, dtype=torch.complex64)

    def half(self):
        return self._diag(0.0, 0.0, np.pi, np.pi)

    def quarter(self):
        return self._diag(0.0, 0.0, np.pi / 2.0, np.pi / 2.0)

    def any(self,phi1, phi2, phi3, phi4):
        return self._diag(phi1, phi2, phi3, phi4)


alpha_deg_list=[0.0,20.0,40.0,60.0]
alpha_list=[deg/180.0*np.pi for deg in alpha_deg_list]
# alpha = 20.0/180.0*np.pi  # basis changing angle
degree_list = [20] # polarization angle
theta_list=[deg/180.0*np.pi for deg in degree_list]

whole_power_list=[]
pol_ang_start_list=[]
pol_ang_diff_list=[]
for alpha in alpha_list:
    for theta in theta_list:
        Eax = np.cos(theta)
        Ebx = 0.0
        Eay = np.sin(theta)
        Eby = 0.0
        E = [[Eax], [Ebx], [Eay], [Eby]]
        E_tensor = torch.tensor(E, dtype=torch.complex64)

        Rot_base = Basis_change()
        R1 = Rot_base(alpha)
        R2 = Rot_base(-alpha)
        ret = wave_plate().quarter()
        # ret = wave_plate().any(np.pi,0,0,0)
        H = R2.mm(ret).mm(R1)

        res = H.mm(E_tensor)
        phi_list = np.linspace(0, 2.0 * np.pi, num=360)
        res_x = res[0][0] * np.cos(phi_list) + res[1][0] * np.sin(phi_list)
        res_y = res[2][0] * np.cos(phi_list) + res[3][0] * np.sin(phi_list)

        whole_power_x = (np.abs(res_x)) ** 2
        whole_power_y = (np.abs(res_y)) ** 2
        whole_power = whole_power_y + whole_power_x
        whole_power_list.append(whole_power)

        # polarization result
        pol_ang_start = np.arctan(np.real(res_y) / np.real(res_x))
        pol_ang_start_list.append(pol_ang_start)
        pol_ang_diff = np.angle(res_y) - np.angle(res_x)
        pol_ang_diff_list.append(pol_ang_diff)

# plot

def theta_to_deg(theta_list):
    deg_list=[theta/np.pi*180 for theta in theta_list]
    return deg_list


fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
for i in range(len(whole_power_list)):
    ax1.plot(theta_to_deg(phi_list), whole_power_list[i], label=f"alpha={alpha_deg_list[i]}")
    ax2.plot(theta_to_deg(phi_list), theta_to_deg(pol_ang_start_list)[i], label=f"alpha={alpha_deg_list[i]}")
    ax3.plot(theta_to_deg(phi_list), theta_to_deg(pol_ang_diff_list)[i], label=f"alpha={alpha_deg_list[i]}")

ax1.legend()
ax2.legend()
ax3.legend()
