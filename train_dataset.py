from torch.utils.data import Dataset
import os
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
class pet_train_dataset(Dataset):
    def __init__(self, train_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_train = []
        self.loaded_slices = {}

        # Load all patient slices
        all_patient_slice_list = []
        for line in open(os.path.join(train_dir, 'train.txt')):
            line = line.strip('\n')
            if line:
                base_filename = line[:-4]
                all_patient_slice_list.append(base_filename)

        # Group slices into lists of 5
        for i in range(0, len(all_patient_slice_list), 5):
            group = all_patient_slice_list[i:i+5]
            self.list_train.append(group)
        self.root_singram = os.path.join(train_dir, '1_sino/')
        self.root_gt_sin = os.path.join(train_dir, 'gt_sino/')
        self.root_img = os.path.join(train_dir, 'gt_img/')
        self.file_len = len(self.list_train)

    def __getitem__(self, index):
        slices_singram = []
        slices_img = []
        slices_gt_singram = []  # List to store gt_sin images
        slice_group = self.list_train[index]

        for slice_name in slice_group:
            if slice_name in self.loaded_slices:
                pet_singram, pet_img = self.loaded_slices[slice_name]
            else:
                # Add the .npy extension to the slice name
                pet_singram_path = os.path.join(self.root_singram, slice_name + '.npy')
                gt_singram_path = os.path.join(self.root_singram, slice_name + '.npy')
                pet_img_path = os.path.join(self.root_img, slice_name + '.png')

                # Load the .npy files
                pet_singram = np.load(pet_singram_path)
                pet_gt_singram = np.load(gt_singram_path)
                pet_img = Image.open(pet_img_path)

                # Convert numpy arrays to torch tensors
                pet_singram = torch.from_numpy(pet_singram).float()
                pet_gt_singram = torch.from_numpy(pet_gt_singram).float()
                pet_img = self.transform(pet_img)

                # Cache the loaded tensors
                self.loaded_slices[slice_name] = (pet_singram, pet_gt_singram, pet_img)

            slices_singram.append(pet_singram)
            slices_gt_singram.append(pet_gt_singram)
            slices_img.append(pet_img)

        # Stack the list of tensors along a new dimension
        pet_singram_volume = torch.stack(slices_singram, dim=0)
        pet_singram_volume = pet_singram_volume.unsqueeze(0)

        pet_gt_singram_volume = torch.stack(slices_gt_singram, dim=0)
        pet_gt_singram_volume = pet_gt_singram_volume.unsqueeze(0)

        pet_img_volume = torch.stack(slices_img, dim=1)
        return pet_singram_volume, pet_img_volume,pet_gt_singram_volume

    def __len__(self):
        return self.file_len
