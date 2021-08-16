import unittest
from reading_real_handwritten_digits_cuda import MNISTDataset, DataLoader, MNISTModel, spatial_size
import torch
import torch.nn.functional as F
import platform
import torch.optim as optim




class TestingHWDigits(unittest.TestCase):
    def setUp(self) -> None:
        self.training_dataset = MNISTDataset()
        self.batch_loader_params = {
            "batch_size": 50,
            "shuffle": True,
            "num_workers": 0 if platform.system() == 'Windows' else 2
        }
        self.dataloader = DataLoader(self.training_dataset, **self.batch_loader_params)
        self.model = MNISTModel()

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
        sample = next(iter(self.training_dataset))
        image = sample['image']
        label = sample['label']
        image_batch = image.unsqueeze(0)
        self.assertEqual(image_batch.shape, torch.Size([1, 1, 28, 28]), 'Your dimensions of your input should be: Batch, Channels, Pixel Width, Pixel Height')
        self.assertEqual(self.model(image_batch).shape, torch.Size([1, 10]), 'Your dimensions of your input should be: Batch, Possible Outcomes')
        self.assertEqual(label.shape[0], image_batch.shape[0], 'Your output should match the batch number of your input')

    def test_one_sample_into_model_to_calculate_first_loss(self):
        sample = next(iter(self.training_dataset))
        image = sample['image']
        labels = sample['label']
        image_batch = image.unsqueeze(0)

        preds = self.model(image_batch)
        loss = F.cross_entropy(preds, labels)
        self.assertGreaterEqual(loss.item(), 0, 'Your loss should be greater than 0')
        self.assertEqual(self.model.conv1.weight.grad, None, 'No grad should exist beofre the backward method is called')
        loss.backward()
        self.assertEqual(self.model.conv1.weight.grad.shape, torch.Size([20, 1, 5, 5]), 'Conv1 shape must be consistant ')
    
    def running_one_sample_repeatedly_looking_for_correct_output(self):
        '''This isn't really a test, but it shows how the model will adapt to the same sample feeding it into itself. Look at the preds matrix, and see the changes during
        each iteration'''
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        sample = next(iter(self.training_dataset))
        image = sample['image']
        labels = sample['label']
        image_batch = image.unsqueeze(0)

        for _ in range(10):
            preds = self.model(image_batch)
            loss = F.cross_entropy(preds, labels)
            loss.backward()
            print(f'{loss.item()=}')

            print(preds, labels)
            print(f"{preds.argmax()=}")
            print(f"{preds.argmax(dim=1)=}")

            optimizer.step()
            optimizer.zero_grad()

    def tearDown(self) -> None:
        pass
if __name__=='__main__':
    unittest.main()
