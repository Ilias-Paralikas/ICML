import torch
from torch.utils.data import Dataset
import os

class UDAFrameDataset(Dataset):
    def __init__(self, 
                 number_of_frames=2,
                 root='../data_handlers/data/CardiacUDA/sliced_data/train',
                 train_augmentations=None):

        self.root = root
        self.number_of_frames = number_of_frames
        self.train_augmentations = train_augmentations
        self.site_folders = [os.path.join(self.root, site) for site in os.listdir(self.root)]
 

        self.data_paths = []
        self.site_folders = [os.path.join(self.root, site) for site in os.listdir(self.root)]
        for site in self.site_folders:
            patients = os.listdir(site)
            for patient in patients:
                patient_folder = os.path.join(site, patient)
                number_of_slices = len(os.listdir(patient_folder))

                slices = [f'slice_{i}' for i in range(number_of_slices)]
                for i in range(number_of_slices-self.number_of_frames):
                    temp_slice_paths = []
                    for j in range(i,i+self.number_of_frames):
                        slice_folder = os.path.join(patient_folder, slices[j])
                        data_file = os.path.join(slice_folder, 'slice.pt')
                        temp_slice_paths.append(data_file)
                    self.data_paths.append(temp_slice_paths)
                    
        self.len = len(self.data_paths)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        frames = []
        for path in self.data_paths[idx]:
            frame = torch.load(path)

            if self.train_augmentations is not None:
                frame = self.train_augmentations(frame)
            frames.append(frame)
        return torch.stack(frames)     
