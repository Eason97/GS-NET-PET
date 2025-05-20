import torch
import torch.nn as nn
import numpy as np
from subnets import Sino_modify_net,F_transform, F_inverse
import sys
import torch.nn.init as init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def conv_3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    layer = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.ReLU())
    # init.xavier_uniform_(layer[0].weight)
    # if layer[0].bias is not None:
    #     nn.init.constant_(layer[0].bias, 0)
    return layer

log_file = open('training_log.txt', 'w')

def check_for_nan(tensor, message):
    if torch.isnan(tensor).any():
        log_file.write(f"NaN detected in {message}\n")
        log_file.close()  # Close the file before stopping
        sys.exit("Training stopped due to NaN values")

def print_stats(tensor, name):
    max_val = tensor.max().item()
    min_val = tensor.min().item()
    mean_val = tensor.mean().item()
    log_file.write(f"{name} - max: {max_val}, min: {min_val}, mean: {mean_val}\n")



def np_to_torch(array):
    return torch.from_numpy(array).float()

def fp_system_torch_batch_3d(image_batch, sys_mat, nxd, nrd, nphi, depth):
    batch_size = image_batch.shape[0]
    reshaped_image_batch = torch.reshape(image_batch, (batch_size * depth, nxd * nxd, 1))
    sys_expanded = sys_mat.unsqueeze(0).expand(batch_size * depth, *sys_mat.shape)
    projected_batch = torch.bmm(sys_expanded, reshaped_image_batch)
    sinograms = torch.reshape(projected_batch, (batch_size, 1, depth, nphi, nrd))
    print_stats(sinograms, "Sinograms")
    check_for_nan(sinograms, "Sinogram Projection")
    return sinograms

def bp_system_torch_batch_3d(sino_batch, sys_mat, nxd, nrd, nphi, depth):
    batch_size = sino_batch.shape[0]
    reshaped_sino_batch = torch.reshape(sino_batch, (batch_size * depth, nrd * nphi, 1))
    sys_transposed = sys_mat.T
    sys_transposed_expanded = sys_transposed.unsqueeze(0).expand(batch_size * depth, *sys_transposed.shape)
    matrix_multiplication = torch.bmm(sys_transposed_expanded, reshaped_sino_batch)
    output_images = torch.reshape(matrix_multiplication, (batch_size, 1, depth, nxd, nxd))
    print_stats(output_images, "Output Images")
    check_for_nan(output_images, "Backprojection Output")
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
            denominator = torch.clamp(denominator, min=1.0e-9)
            check_for_nan(denominator, "Denominator")

            sino_for_reconstruction = self.sino_modification(sino_for_reconstruction)
            print_stats(sino_for_reconstruction, "Sino Modification Output")
            check_for_nan(sino_for_reconstruction, "Sino Modification Output")

            sino_domain = sino_for_reconstruction / (denominator)
            print_stats(sino_domain, "Sino Domain")
            check_for_nan(sino_domain, "Sino Domain")

            correction = bp_system_torch_batch_3d(sino_domain, sys_mat, self.nxd, self.nrd, self.nphi, 5)
            print_stats(correction, "Correction")
            check_for_nan(correction, "Correction")

            clamped_rho_param = torch.clamp(self.rho_param, min=0.98, max=1.02)
            A = clamped_rho_param
            B = clamped_rho_param * beta + clamped_rho_param * zeta - 1
            B = torch.clamp(B, min=-1e12, max=1e12)
            print_stats(B, "B values")

            C = x * correction
            C = torch.clamp(C, min=-1e12, max=1e12)
            print_stats(C, "C values")

            discriminant_expression = B ** 2 + 4 * A * C
            print_stats(discriminant_expression, "Discriminant Expression Before Clamp")
            discriminant_expression_clamped = torch.clamp(discriminant_expression, min=0)
            print_stats(discriminant_expression_clamped, "Discriminant Expression Clamped")

            discriminant = torch.sqrt(discriminant_expression_clamped)
            print_stats(discriminant, "Discriminant")

            x = (-B + discriminant) / (2 * A)
            print_stats(x, "Updated X")
            check_for_nan(x, "Updated X")

            sin_domain_loss = gt_sin - sino_for_reconstruction
            print_stats(sin_domain_loss, "Sin Domain Loss")
            check_for_nan(sin_domain_loss, "Sin Domain Loss")

        return x, sin_domain_loss


class Z_update(nn.Module):
    def __init__(self):
        super(Z_update, self).__init__()
        self.soft_thr = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
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
        print_stats(z_in, "Z_in after Conv_in")

        z_forward = self.F_transform(z_in)
        print_stats(z_forward, "Z_forward after F_transform")

        soft = torch.mul(torch.sign(z_forward), torch.clamp(torch.abs(z_forward) - self.soft_thr, min=0))
        print_stats(soft, "Soft Thresholding")

        z_backward = self.F_inverse(soft)
        print_stats(z_backward, "Z_backward after F_inverse")

        z_out = self.conv_out(z_backward)
        print_stats(z_out, "Z_out after Conv_out")

        wtf_input = torch.rand((1, 32, 5, 144, 144), device=device)
        print_stats(wtf_input, "WTF Input")

        wtf_forward = self.F_transform(wtf_input)
        print_stats(wtf_forward, "WTF Forward after F_transform")

        wtf_forward_backward = self.F_inverse(wtf_forward)
        print_stats(wtf_forward_backward, "WTF Forward Backward after F_inverse")

        consist_z = wtf_input - wtf_forward_backward
        print_stats(consist_z, "Consistency Check Z")

        return z_out, consist_z, total_frobenius_norm_squared


class Beta_update(nn.Module):
    def __init__(self):
        super(Beta_update, self).__init__()
        self.eta = nn.Parameter(torch.Tensor([0.001]), requires_grad=True)

    def forward(self, x, zeta, beta):
        beta = beta + (x - zeta) * self.eta
        print_stats(beta, "Updated Beta")
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


if __name__ == '__main__':
    A = np.load('/home/eason/Videos/20240618_brainweb_dataset(100to1000)/system_A.npy')
    image_size = 144
    sinogram_height = 180
    sinogram_width =205
    depth=5
    batch_size=2
    A_torch=np_to_torch(A).to(device)

    sinogram_torch=torch.randn(2,1,5,180,205).to(device)
    ADMM_method=ADMM_NET(144, 205, 180, 2,10).to(device)
    reconstructed_image,sino_loss,all_consist_losses,all_regularization_losses=ADMM_method(sinogram_torch, A_torch, sinogram_torch)
    print(reconstructed_image.shape)

