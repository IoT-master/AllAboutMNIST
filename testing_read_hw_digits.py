import unittest
from reading_real_handwritten_digits_cuda import MNISTDataset, DataLoader, MNISTModel, spatial_size
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
        self.assertEqual(sample['label'].shape, torch.Size([1]), 'Dimensions are not correct')

    def test_basic_dataloader(self):
        sample = iter(self.dataloader)
        temp = sample.next()
        self.assertEqual(temp['label'].shape, torch.Size([50, 1]), 'Label is broken')
        self.assertEqual(temp['image'].shape, torch.Size([50, 1, 28, 28]), 'Dataloader input is broken')

    def test_spatial_size(self):
        self.assertEqual(spatial_size(28, 5), 24, 'Spatial Size function is broken')

    def test_one_sample_into_model(self):
            training_dataset = MNISTDataset()
            batch_loader_params = {
                "batch_size": 50,
                "shuffle": True,
                "num_workers": 0 if platform.system() == 'Windows' else 2
            }
            dataloader = DataLoader(training_dataset, **batch_loader_params)
            model = MNISTModel()
            sample = next(iter(training_dataset))
            image = sample['image']
            label = sample['label']
            image_batch = image.unsqueeze(0)
            self.assertEqual(image_batch.shape, torch.Size([1, 1, 28, 28]), 'Your dimensions of your input should be: Batch, Channels, Pixel Width, Pixel Height')
            self.assertEqual(model(image_batch).shape, torch.Size([1, 10]), 'Your dimensions of your input should be: Batch, Possible Outcomes')

    def tearDown(self) -> None:
        pass
if __name__=='__main__':
    unittest.main()
