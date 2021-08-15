import unittest
from reading_real_handwritten_digits_cuda import MNISTDataset, DataLoader
import torch
import platform




class TestingHWDigits(unittest.TestCase):
    def setUp(self) -> None:
        training_dataset = MNISTDataset()
        self.batch_loader_params = {
            "batch_size": 50,
            "shuffle": True,
            "num_workers": 0 if platform.system() == 'Windows' else 2
        }
        self.dataloader = DataLoader(training_dataset, **self.batch_loader_params)

    def test_cuda_available(self):
        self.assertEqual(torch.cuda.is_available(), True, 'CUDA is not working or MIA')

    def test_function(self):
        dataset = MNISTDataset()
        sample = dataset.__getitem__(0)
        self.assertEqual(sample['image'].shape, torch.Size([1, 28, 28]), 'Dimensions are not correct')
        self.assertEqual(sample['label'], 4, 'Dimensions are not correct')

    def test_basic_dataloader(self):
        sample = iter(self.dataloader)
        temp = sample.next()
        self.assertEqual(temp['label'].shape, torch.Size([50]), 'Label is broken')
        self.assertEqual(temp['image'].shape, torch.Size([50, 1, 28, 28]), 'Dataloader input is broken')

if __name__=='__main__':
    unittest.main()
