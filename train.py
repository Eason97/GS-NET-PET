import torch
import argparse
from admm_mapem import ADMM_NET
from train_dataset import pet_train_dataset
from test_dataset import pet_test_dataset
from torch.utils.data import DataLoader
import os
from utils import to_psnr,to_ssim_skimage,to_rmse
from tensorboardX import SummaryWriter
from torchvision.utils import save_image as imwrite
import torch.nn.functional as F
import numpy as np
import random
from pytorch_msssim import msssim

# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='ADMM-ISTA')
parser.add_argument('-learning_rate', help='Set the learning rate', default=3e-5, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=8, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=1000, type=int)
parser.add_argument('--train_dataset', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--model_save_dir', type=str, default='./check_points')
parser.add_argument('--log_dir', type=str, default=None)
# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=8, type=int)
args = parser.parse_args()

# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch = args.train_epoch
# /home/eason/Videos/20240804_lqpet50_dataset
train_dataset_dir = os.path.join('/home/eason/Videos/20240804_lqpet50_dataset/')
# --- test --- #
test_dataset_dir = os.path.join('/home/eason/Videos/20240804_lqpet50_dataset/')
predict_result = args.predict_result
test_batch_size = args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
output_dir = os.path.join('check_points/')

def np_to_torch(array):
    return torch.from_numpy(array).float()
# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Set the random seed for reproducibility --- #
def set_random_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # For CUDA-enabled GPUs
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True  # For consistent results on the CuDNN backend
    torch.backends.cudnn.benchmark = False  # For consistent results on the CuDNN backend
seed_value = 42
set_random_seed(seed_value)

# --- Define the network --- #
# /home/eason/Videos/20240804_lqpet50_dataset/udpet_system_A.npy
A = np.load('/home/eason/Videos/20240804_lqpet50_dataset/udpet_system_A.npy')
sys_mat= np_to_torch(A).to(device)
sys_mat.requires_grad = False

GNet = ADMM_NET(144, 205, 180, 1,10)
print('GNet parameters:', sum(param.numel() for param in GNet.parameters()))

# --- Build optimizer --- #
G_optimizer = torch.optim.Adam(GNet.parameters(), lr=3e-5)

# --- Load training data --- #
train_dataset = pet_train_dataset(train_dataset_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True,num_workers=10, pin_memory=True)
# --- Load testing data --- #
test_dataset = pet_test_dataset(test_dataset_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True,num_workers=10, pin_memory=True)

GNet= GNet.to(device)
# Load the model checkpoint
# checkpoint = torch.load('gsnet_100k_epoch320.pkl', map_location=device)
# GNet.load_state_dict(checkpoint)


writer = SummaryWriter()
# --- Start training --- #
iteration = 0
best_psnr = 0
train_loss_list = []

# Open a file to record the metrics
if not os.path.exists('metrics'):
    os.makedirs('metrics')
file_path = os.path.join('metrics', 'training_metrics.txt')
with open(file_path, 'w') as f:
    f.write('Epoch,Training Loss,Testing Loss,Testing PSNR,Testing SSIM,Testing RMSE\n')

def check_for_nans(model):
    for name, layer in model.named_modules():
        for param_name, param in layer.named_parameters():
            if torch.isnan(param).any():  # Check if any parameter in the layer has NaN
                return True, f"{name} ({param_name})"
    return False, None


initial_consist_weight = 0.0001
initial_frobenius_weight = 0.0000001
adjusted_consist_weight = initial_consist_weight
adjusted_frobenius_weight = initial_frobenius_weight


def update_weights(sinogram_loss, consist_loss, frobenius_loss):
    global adjusted_consist_weight, adjusted_frobenius_weight

    # Detach the tensors from the computation graph, move them to CPU, and convert to numpy
    sinogram_loss_cpu = sinogram_loss.detach().cpu().numpy()
    consist_loss_cpu = consist_loss.detach().cpu().numpy()
    frobenius_loss_cpu = frobenius_loss.detach().cpu().numpy()

    # Calculate the order of magnitude of sinogram_loss
    magnitude_sinogram = 10 ** np.floor(np.log10(sinogram_loss_cpu))

    # Set the weights for consist_loss and frobenius_loss
    adjusted_consist_weight = 0.1 * magnitude_sinogram / consist_loss_cpu
    adjusted_frobenius_weight = 0.1 * magnitude_sinogram / frobenius_loss_cpu

    # Ensure the adjusted weights do not exceed the initial weights
    adjusted_consist_weight = min(adjusted_consist_weight, initial_consist_weight)
    adjusted_frobenius_weight = min(adjusted_frobenius_weight, initial_frobenius_weight)



layer_num=10
msssim_loss = msssim
for epoch in range(train_epoch):
    GNet.train()
    # scheduler.step()
    epoch_loss = 0
    for batch_idx, (sin, img, gt_sin) in enumerate(train_loader):
        iteration += 1
        sin = sin.to(device)
        img = img.to(device)
        gt_sin = gt_sin.to(device)
        rec_pet, sin_loss, all_consist_losses,all_frobenius_losses = GNet(sin, sys_mat, gt_sin)

        rec_pet = rec_pet.view(-1, 1, 144, 144)
        img = img.view(-1, 1, 144, 144)
        mse_loss = F.mse_loss(rec_pet, img)

        sinogram_loss = torch.mean(torch.pow(sin_loss[0], 2))
        for k in range(layer_num - 1):
            sinogram_loss += torch.mean(torch.pow(sin_loss[k + 1], 2))

        # Clear gradients
        GNet.zero_grad()

        consist_loss = torch.mean(torch.pow(all_consist_losses[0], 2))
        for k in range(layer_num - 1):
            consist_loss += torch.mean(torch.pow(all_consist_losses[k + 1], 2))
        consist_loss = 0.1 * consist_loss

        frobenius_loss = torch.mean(all_frobenius_losses[0])
        for k in range(layer_num - 1):
            frobenius_loss += torch.mean(all_frobenius_losses[k + 1])
        frobenius_loss = 0.25 * frobenius_loss

        msssim_loss_ = -msssim_loss(rec_pet, img, normalize=True)
        #0.2 * msssim_loss_+
        total_loss = mse_loss + sinogram_loss + adjusted_consist_weight * consist_loss + adjusted_frobenius_weight * frobenius_loss
        total_loss.backward()


        nan_found, problematic_layer = check_for_nans(GNet)
        if nan_found:
            print(f"NaN detected in parameters of the layer: {problematic_layer}")
            break

        torch.nn.utils.clip_grad_norm_(GNet.parameters(), max_norm=2.0)
        G_optimizer.step()

        # Log losses and total loss for monitoring
        writer.add_scalars('Losses', {
            'mse_loss': mse_loss.item(),
            'sinogram_loss': sinogram_loss.item(),
            'consist_loss': adjusted_consist_weight * consist_loss.item(),
            'frobenius_loss':adjusted_frobenius_weight * frobenius_loss.item(),
            'Total Loss': total_loss.item()}, iteration)

        # Accumulate total loss for the epoch
        epoch_loss += total_loss.item()

    # Calculate and log the average loss for the epoch
    average_epoch_loss = epoch_loss / len(train_loader)
    train_loss_list.append(average_epoch_loss)
    update_weights(sinogram_loss, consist_loss, frobenius_loss)



    if epoch % 1 == 0:
        print('we are testing on epoch: ' + str(epoch))
        with torch.no_grad():
            psnr_list = []
            ssim_list = []
            rmse_list = []
            test_loss = 0
            GNet.eval()
            for batch_idx, (sin,img,gt_sin,name) in enumerate(test_loader):
                sin = sin.to(device)
                img = img.to(device)
                gt_sin=gt_sin.to(device)
                rec_pet,_,_,_= GNet(sin, sys_mat,gt_sin)

                rec_pet = rec_pet.view(-1,1,144,144)
                img = img.view(-1,1,144,144)

                mse_test_loss = F.mse_loss(rec_pet, img)
                test_loss += mse_test_loss.item()

                psnr_list.extend(to_psnr(rec_pet, img))
                ssim_list.extend(to_ssim_skimage(rec_pet, img))
                rmse_list.extend(to_rmse(rec_pet, img))

            avr_psnr = sum(psnr_list) / len(psnr_list)
            avr_ssim = sum(ssim_list) / len(ssim_list)
            avr_rmse = sum(rmse_list) / len(rmse_list)
            average_test_loss = test_loss / len(test_loader)
            writer.add_scalars('psnr', {'testing psnr': avr_psnr}, epoch)
            writer.add_scalars('ssim', {'testing ssim': avr_ssim}, epoch)
            writer.add_scalars('rmse', {'testing rmse': avr_rmse}, epoch)

            file_path = os.path.join('metrics', 'training_metrics.txt')
            with open(file_path, 'a') as f:
                f.write(f"{epoch},{epoch_loss / len(train_loader)},{average_test_loss},{avr_psnr},{avr_ssim},{avr_rmse}\n")

            # Save the model only if the current PSNR is greater than the previous best PSNR
            if avr_psnr > best_psnr or avr_ssim>0.75:
                print(f"New best ssim achieved: {avr_psnr,avr_ssim}. Saving model...")
                torch.save(GNet.state_dict(), os.path.join(args.model_save_dir, 'gsnet_100k_epoch' + str(epoch) + '.pkl'))
                best_psnr = avr_psnr