import torch
from torchvision import transforms
from pathlib import Path
import pandas as pd
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import platform

MNIST_DIR = "MNISTDataSet"

testing_df = pd.read_csv(Path(MNIST_DIR).joinpath('test.csv'))

mnist_dir = Path.cwd().joinpath(MNIST_DIR)



def spatial_size(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0):
    # https://cs231n.github.io/convolutional-networks/
    spatial_size = (input_size - kernel_size + 2 * padding)/stride + 1
    assert spatial_size % 1 == 0
    assert spatial_size > 0
    print(
        f'You will have {spatial_size**2:.0f} feature maps if square dimensions')
    return int(spatial_size)

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

class MNISTDataset(Dataset):

    def __init__(self, transform=None) -> None:
        self.csv_path = Path(MNIST_DIR).joinpath('train.csv')
        self.training_path = Path(MNIST_DIR).joinpath('train')
        super().__init__()
        self.training_df = pd.read_csv(self.csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.training_df)

    def __getitem__(self, index) -> dict:
        path_to_image = Path(self.training_path).joinpath(self.training_df['filename'].iloc[index])
        image = cv2.imread(str(path_to_image), cv2.IMREAD_GRAYSCALE)
        torched_image = torch.Tensor(image)
        image_x, image_y = torched_image.shape
        cv2_converted_tensor = torched_image.view(-1, image_x, image_y)
        normalized_filter = transforms.Normalize(torch.mean(cv2_converted_tensor), torch.std(cv2_converted_tensor))
        data = normalized_filter(cv2_converted_tensor)
        label = self.training_df['label'].iloc[index]
        sample = {
            'image': data,
            'label': label
        }

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

training_dataset = MNISTDataset()
batch_loader_params = {
    "batch_size": 50,
    "shuffle": True,
    "num_workers": 0 if platform.system() == 'Windows' else 2
}
dataloader = DataLoader(training_dataset, **batch_loader_params)

if __name__ == "__main__":
    print(spatial_size(28, 3))