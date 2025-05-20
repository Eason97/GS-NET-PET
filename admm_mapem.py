import torch
import torch.nn as nn
from subnets import Sino_modify_net,F_transform, F_inverse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def conv_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    layer = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU())
    return layer

def np_to_torch(array):
    return torch.from_numpy(array).float()

def fp_system_torch_batch_3d(image_batch, sys_mat, nxd, nrd, nphi, depth):
    batch_size = image_batch.shape[0]
    reshaped_image_batch = torch.reshape(image_batch, (batch_size * depth, nxd * nxd, 1))
    sys_expanded = sys_mat.unsqueeze(0).expand(batch_size * depth, *sys_mat.shape)
    projected_batch = torch.bmm(sys_expanded, reshaped_image_batch)
    sinograms = torch.reshape(projected_batch, (batch_size, 1, depth, nphi, nrd))
    return sinograms

def bp_system_torch_batch_3d(sino_batch, sys_mat, nxd, nrd, nphi, depth):
    batch_size = sino_batch.shape[0]
    reshaped_sino_batch = torch.reshape(sino_batch, (batch_size * depth, nrd * nphi, 1))
    sys_transposed = sys_mat.T
    sys_transposed_expanded = sys_transposed.unsqueeze(0).expand(batch_size * depth, *sys_transposed.shape)
    matrix_multiplication = torch.bmm(sys_transposed_expanded, reshaped_sino_batch)
    output_images = torch.reshape(matrix_multiplication, (batch_size, 1, depth, nxd, nxd))
    return output_images

class X_update(nn.Module):
    def __init__(self, nxd, nrd, nphi, num_its):
        super(X_update, self).__init__()
        self.nxd = nxd
        self.nrd = nrd
        self.nphi = nphi
        self.num_its = num_its
        self.rho_param = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.sino_modification = Sino_modify_net()

    def forward(self, sino_for_reconstruction, gt_sin, sys_mat, x, zeta, beta):
        sin_domain_loss = 0
        for it in range(self.num_its):
            denominator = fp_system_torch_batch_3d(x, sys_mat, self.nxd, self.nrd, self.nphi, 5) + 1.0e-9
            sino_for_reconstruction = self.sino_modification(sino_for_reconstruction)
            sino_domain = sino_for_reconstruction / (denominator)
            correction = bp_system_torch_batch_3d(sino_domain, sys_mat, self.nxd, self.nrd, self.nphi, 5)
            clamped_rho_param = torch.clamp(self.rho_param, min=0.98, max=1.02)
            A = clamped_rho_param
            B = clamped_rho_param * beta + clamped_rho_param * zeta - 1
            C = x * correction
            discriminant_expression = B ** 2 + 4 * A * C
            discriminant_expression_clamped = torch.clamp(discriminant_expression, min=0)
            discriminant = torch.sqrt(discriminant_expression_clamped)
            x = (-B + discriminant) / (2 * A)
            sin_domain_loss = gt_sin - sino_for_reconstruction

        return x, sin_domain_loss


class Z_update(nn.Module):
    def __init__(self):
        super(Z_update, self).__init__()
        self.soft_thr = nn.Parameter(torch.Tensor([0.001]), requires_grad=True)
        self.conv_in = conv_3d(1, 32)
        self.F_transform = F_transform()
        self.F_inverse = F_inverse()
        self.conv_out = conv_3d(32, 1)

    def forward(self, z, beta):
        weight_f_trans_0 = self.F_transform.layer1[0].weight
        weight_f_trans_3 = self.F_transform.layer1[2].weight
        weight_f_inv_0 = self.F_inverse.layer1[0].weight
        weight_f_inv_3 = self.F_inverse.layer1[2].weight

        # Calculating squared Frobenius norms
        norm_f_trans_0 = torch.norm(weight_f_trans_0, p='fro').pow(2)
        norm_f_trans_3 = torch.norm(weight_f_trans_3, p='fro').pow(2)
        norm_f_inv_0 = torch.norm(weight_f_inv_0, p='fro').pow(2)
        norm_f_inv_3 = torch.norm(weight_f_inv_3, p='fro').pow(2)

        total_frobenius_norm_squared = norm_f_trans_0 + norm_f_trans_3 + norm_f_inv_0 + norm_f_inv_3

        z_in = self.conv_in(z + beta)
        z_forward = self.F_transform(z_in)
        soft = torch.mul(torch.sign(z_forward), torch.clamp(torch.abs(z_forward) - self.soft_thr, min=0))
        z_backward = self.F_inverse(soft)
        z_out = self.conv_out(z_backward)
        consist_z = z_forward - z_backward

        return z_out, consist_z, total_frobenius_norm_squared


class Beta_update(nn.Module):
    def __init__(self):
        super(Beta_update, self).__init__()
        self.eta = nn.Parameter(torch.Tensor([0.001]), requires_grad=True)

    def forward(self, x, zeta, beta):
        beta = beta + (x - zeta) * self.eta
        return beta


class ADMM_NET_Block(nn.Module):
    def __init__(self, nxd, nrd, nphi, num_its):
        super(ADMM_NET_Block, self).__init__()
        self.x_update = X_update(nxd, nrd, nphi, num_its)
        self.z_update = Z_update()
        self.beta_update = Beta_update()
        self.nxd = nxd
        self.nrd = nrd
        self.nphi = nphi
        self.num_its = num_its
        self.device = device

    def forward(self, sino_for_reconstruction, sys_mat, gt_sin,x=None, z=None, beta=None):
        batch_size = sino_for_reconstruction.shape[0]
        if x is None:
            x = torch.zeros((batch_size, 1, 5, self.nxd, self.nxd), device=self.device)
        if z is None:
            z = torch.zeros((batch_size, 1, 5, self.nxd, self.nxd), device=self.device)
        if beta is None:
            beta = torch.zeros((batch_size, 1, 5, self.nxd, self.nxd), device=self.device)

        x,sino_loss = self.x_update(sino_for_reconstruction, gt_sin, sys_mat, x, z, beta)
        z, consist_loss, frobenius_loss = self.z_update(x, beta)
        beta = self.beta_update(x, z, beta)
        return x, z, beta, sino_loss, consist_loss, frobenius_loss

class ADMM_NET(nn.Module):
    def __init__(self, nxd, nrd, nphi, num_its, num_blocks):
        super(ADMM_NET, self).__init__()
        self.blocks = nn.ModuleList([ADMM_NET_Block(nxd, nrd, nphi, num_its) for _ in range(num_blocks)])

    def forward(self, sino_for_reconstruction, sys_mat,sino_gt):
        x, z, beta = None, None, None
        all_sino_losses=[]
        all_consist_losses=[]
        all_frobenius_losses=[]
        for block in self.blocks:
            x, z, beta, sino_loss, consist_loss, frobenius_loss = block(sino_for_reconstruction, sys_mat, sino_gt,x, z, beta)
            all_sino_losses.append(sino_loss)
            all_consist_losses.append(consist_loss)
            all_frobenius_losses.append(frobenius_loss)
        return x, all_sino_losses,all_consist_losses,all_frobenius_losses