from torch.utils.data import Dataset
import os

import nibabel as nib
class SiteDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.patient_paths = {}
        files = os.listdir(self.root)
        for f in files:
            patient_num =f.split('_')[0]
            if patient_num not in self.patient_paths.keys():
                self.patient_paths[patient_num] = {}
            type =f.split('_')[1]
            if type =='image.nii.gz':
                self.patient_paths[patient_num]['image'] = os.path.join(self.root,f)
            elif type =='label.nii.gz':
                self.patient_paths[patient_num]['label'] = os.path.join(self.root,f)
        self.len = len(self.patient_paths)
   
    def __len__(self):
        return self.len
                
    def __getitem__(self, idx):
        patient_num = list(self.patient_paths.keys())[idx]
        patient_path = self.patient_paths[patient_num]
        image_path = patient_path['image']
        image = nib.load(image_path).get_fdata()
        
        try:
            label_path = patient_path['label']
            label = nib.load(label_path).get_fdata()
        except:
            label = None
        path = os.path.join(self.root,patient_num)
        return image,label, path
