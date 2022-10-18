import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import os
import utils
import numpy as np


class GeneralDataset(Dataset):
    def __init__(self,
                 root_dir,
                 path_to_df,
                 classes,
                 sample_rate,
                 dataset_name='arabic',
                 transform=None):
        """
        :param root_dir: Path to root dataset path_to/dataset/
        :param path_to_df: Path to dataframe
        :param classes: List of classes
        :param noise_path: Path to background noise
        """
        super(GeneralDataset, self).__init__()
        self.root_dir = root_dir
        self.path_to_df = path_to_df
        self.df = pd.read_csv(path_to_df)

        self.classes = classes
        self.n_class = len(classes)
        self.sample_rate = sample_rate
        self.transform = transform
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row_data = self.df.iloc[idx]
        wav, _ = utils.load_audio(os.path.join(self.root_dir, str(row_data['file'])), self.sample_rate)
        # print(row_data)
        if self.dataset_name == 'arabic':
            target = utils.label2index(self.classes, int(row_data['speaker']))
        elif self.dataset_name.startswith('gg-speech'):
            target = utils.label2index(self.classes, row_data['speaker'])

        data = {
            'samples': wav,
            'word': row_data['word'],
            'sample_rate': self.sample_rate,
            'target': target,
            'path': row_data['file']
        }

        if self.transform is not None:
            data = self.transform(data)

        return data


class BackgroundNoiseDataset(Dataset):
    def __init__(self, path, transform, sample_rate, sample_length=1):
        noise_files = [file for file in os.listdir(path) if file.endswith('.wav')]
        samples = []
        for f in noise_files:
            noise_path = os.path.join(path, f)
            sample, sample_rate = utils.load_audio(noise_path, sample_rate)
            samples.append(sample)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r * c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {
            'samples': self.samples[index],
            'sample_rate': self.sample_rate,
            'target': 1,
            'path': self.path
        }

        if self.transform is not None:
            data = self.transform(data)
        return data

# debug
# if __name__ == '__main__':
#     path = 'F:\\Datasets\\keyword-spotting\\abdulkaderghandoura-arabic-speech-commands-dataset-72f3438\\dataset\\backward\\00000001_NO_01.wav'
#     wav, sample_rate = torchaudio.load(path)
#     wav = torch.squeeze(wav)
#     print(wav.size())
