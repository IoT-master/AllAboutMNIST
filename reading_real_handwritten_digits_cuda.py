from pandas.io.pytables import IndexCol
import torch
from torchvision import transforms
from pathlib import Path
import pandas as pd

MNIST_DIR = "MNISTDataSet"

mnist_dir = Path.cwd().joinpath(MNIST_DIR)
train_filenames = ['/'.join(i.relative_to(Path.cwd()).parts) for i in mnist_dir.joinpath('train').glob('**/*.png')]
test_filenames = ['/'.join(i.relative_to(Path.cwd()).parts) for i in mnist_dir.joinpath('test').glob('**/*.png')]
print(train_filenames[:10])
print(test_filenames[:10])

df = pd.read_csv(Path(MNIST_DIR).joinpath('test.csv'), index_col='filename')
print(df[:10])