from torch.utils.data import Dataset
import torch
import os

from .labels import remap_sample

class UDATestDataset(Dataset):
    def __init__(self,
                 root ='../data_handlers/data/CardiacUDA/test',
                 num_classes=5,
                 train_channels=None):
        super().__init__()

        self.get_folder_path = lambda x : [os.path.join(x,f) for f in os.listdir(x)]
        self.root = root
        self.num_classes = num_classes
        self.train_channels = train_channels if train_channels is not None else list(range(num_classes))
        self.data_paths= []

        self.patient_paths= self.get_folder_path(self.root)
        for p in self.patient_paths:
            slice_paths = self.get_folder_path(p)
            for s in slice_paths:
                data_path  =os.path.join(s,'slice.pt')
                label_path = os.path.join(s,'label.pt')
                self.data_paths.append((data_path,label_path))

        self.len =len(self.data_paths)

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        data_path,label_path =self.data_paths[idx]
        data = torch.load(data_path, weights_only=True).clone()
        label = torch.load(label_path, weights_only=True).clone().float()

        # pad to num_classes channels if the saved label has fewer
        if label.shape[0] < self.num_classes:
            pad = torch.zeros(self.num_classes - label.shape[0], *label.shape[1:])
            label = torch.cat([label, pad], dim=0)

        label = remap_sample(label, self.train_channels)

        return data, label



