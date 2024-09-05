import json
import os
from PIL import Image
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold

class CephalometricDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        with open(json_file) as f:
            self.annotations = json.load(f)
        self.root_dir = root_dir
        self.transform = transform
        self.image_info = self.annotations['images']
        self.annotations_info = {ann['image_id']: ann for ann in self.annotations['annotations']}

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_info = self.image_info[idx]
        img_name = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_name).convert('RGB')
        ann = self.annotations_info.get(img_info['id'], {})
        keypoints = ann.get('keypoints', [])
        keypoints = [(keypoints[i], keypoints[i+1]) for i in range(0, len(keypoints), 3)]
        
        sample = {'image': image, 'landmarks': keypoints}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

def get_kfold_data(dataset, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []
    for train_idx, val_idx in kf.split(dataset):
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        splits.append((train_set, val_set))
    return splits
