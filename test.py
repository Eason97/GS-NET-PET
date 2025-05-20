import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import os
from admm_mapem import ADMM_NET
from test_dataset import pet_test_dataset
import re
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def np_to_torch(array):
    return torch.from_numpy(array).float()

# --- Define the network --- #
A = np.load('/home/eason/Videos/20240804_lqpet50_dataset/udpet_system_A.npy')
sys_mat= np_to_torch(A).to(device)
sys_mat.requires_grad = False

# Load model
model_path = 'gsnet_100k_epoch300.pkl'
model= ADMM_NET(144, 205, 180, 1,10).to(device)
model.load_state_dict(torch.load(model_path))

# Load test dataset
test_dataset_dir = '/home/eason/Videos/20240804_lqpet50_dataset/'
test_dataset = pet_test_dataset(test_dataset_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


# Directory to save images
output_dir = '07_gsnet_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def generate_filename(original_name, index):
    # Extract the base name without extension
    base_name = os.path.splitext(original_name)[0]
    # Find the last number in the base name
    match = re.search(r'\d+$', base_name)
    if match:
        # Increment the number by index
        number = int(match.group()) + index
        new_name = re.sub(r'\d+$', f"{number:08d}", base_name)  # Pad with zeros to 8 digits
        return new_name + '.png'
    return f"{base_name}_{index:08}.png"


model.eval()
with torch.no_grad():
    for i, (pet_singram_volume, pet_img_volume, pet_gt_singram_volume, name) in enumerate(test_loader):
        pet_singram_volume = pet_singram_volume.to(device)
        pet_gt_singram_volume=pet_gt_singram_volume.to(device)
        output, _ , _, _= model(pet_singram_volume,sys_mat,pet_gt_singram_volume)
        output = output.squeeze()
        for j in range(output.shape[0]):
            img_tensor = output[j].unsqueeze(0).unsqueeze(0)
            filename = generate_filename(name[0], j)
            utils.save_image(img_tensor, os.path.join(output_dir, filename), range=(0, 1))
print("Testing complete. Output images are saved.")
