import joblib
import torch
import numpy as np

class BatchData(torch.utils.data.Dataset):
    def __init__(self, path):
        self.x, self.label, self.cat = joblib.load(path)
        print(path, 'num_samples', len(self.x))
        assert len(self.x) == len(self.label) == len(self.cat), "Batch size not equal!"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].T.astype(np.float32), self.cat[idx].astype(np.float32), self.label[idx].astype(np.int)


def get_loader(batch_size, path, shuffle, num_workers):
    dataset = BatchData(path)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, drop_last=True)
