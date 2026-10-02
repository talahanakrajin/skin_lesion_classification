import os
from PIL import Image
from torch.utils.data import Dataset

class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(df['dx'].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[row['dx']]
        if self.transform:
            image = self.transform(image)
        return image, label
           
    