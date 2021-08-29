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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA detected")
else:
    device = torch.device("cpu")
    print("GPU NOT detected!")

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
        path_to_image = Path(self.training_path).joinpath(
            self.training_df['filename'].iloc[index])
        image = cv2.imread(str(path_to_image), cv2.IMREAD_GRAYSCALE)
        torched_image = torch.Tensor(image)
        image_x, image_y = torched_image.shape
        cv2_converted_tensor = torched_image.view(-1, image_x, image_y)
        normalized_filter = transforms.Normalize(torch.mean(
            cv2_converted_tensor), torch.std(cv2_converted_tensor))
        data = normalized_filter(cv2_converted_tensor)
        label = self.training_df['label'].iloc[index]
        sample = {
            'image': data,
            'label': torch.tensor(label)
        }

        if self.transform:
            pass

        return sample


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        # [b, 1, 28, 28] ==> [b, 20, 24, 24]
        x = F.relu(self.conv1(x))
        # [b, 20, 24, 24] ==> [b, 20, 12, 12]
        x = F.max_pool2d(x, 2, 2)
        # [b, 20, 12, 12] ==> [b, 50, 8, 8]
        x = F.relu(self.conv2(x))
        # [b, 50, 8, 8] ==> [b, 50, 4, 4]
        x = F.max_pool2d(x, 2, 2)
        # [b, 50, 4, 4] ==> [b, 50*4*4]
        x = x.view(-1, 4*4*50)
        # [b, 50*4*4] ==> [b, 60]
        x = F.relu(self.fc1(x))
        # [b, 60] ==> [b, 10]
        x = self.fc2(x)
        # F.log_softmax(x, dim=1)
        return x

if __name__ == "__main__":
    training_dataset = MNISTDataset()
    model = MNISTModel().to(device)
    batch_loader_params = {
        "batch_size": len(training_dataset),
        "shuffle": True,
        "num_workers": 1
    }
    training_batches = DataLoader(training_dataset, **batch_loader_params)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    total_loss = 0
    total_correct = 0

    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    for epoch in range(5):
        for batch in training_batches:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            preds = model(images)  # Pass Batch
            loss = F.cross_entropy(preds, labels)  # Calculate Loss

            optimizer.zero_grad()  # PyTorch Adds to this, so you have to zero it out after one batch
            loss.backward()  # Calculate Gradients
            optimizer.step()  # Update Weights

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        print("epoch:", epoch, "total_correct:",
            total_correct, "loss:", total_loss)