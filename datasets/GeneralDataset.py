import random

import pandas as pd
from torch.utils.data import Dataset
import os
import utils
import numpy as np
import soundfile
import glob
from scipy import signal


class GeneralDataset(Dataset):
    def __init__(self,
                 root_dir,
                 path_to_df,
                 classes,
                 sample_rate,
                 dataset_name='arabic',
                 stage=1,
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
        self.stage = stage

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row_data = self.df.iloc[idx]
        wav, _ = utils.load_audio(os.path.join(
            self.root_dir, str(row_data['file'])), self.sample_rate)
        # print(row_data)
        target = utils.label2index(self.classes, str(row_data['speaker']))

        data = {
            'samples': wav,
            'sample_rate': self.sample_rate,
            'target': target,
            'path': row_data['file']
        }

        if self.stage == 1:
            data['word'] = row_data['word'],

        if self.transform is not None:
            data = self.transform(data)

        return data


class BackgroundNoiseDataset(Dataset):
    def __init__(self, path, transform, sample_rate, sample_length=1):
        noise_files = [file for file in os.listdir(
            path) if file.endswith('.wav')]
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


class AugmentationDataset(object):
    def __init__(self, musan_path, rir_path):
        self.musan_path = musan_path
        self.rir_path = rir_path
        # self.duration = duration
        # self.sample_rate = sample_rate
        # self.add_sample = add_sample
        self.noise_types = ['noise', 'speech', 'music']
        self.noises_nr = {'noise': [0, 15],
                          'speech': [13, 20], 'music': [5, 15]}
        self.num_noise = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}
        self.noise_list = {}

        # / or \\
        augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-3] not in self.noise_list:
                self.noise_list[file.split('/')[-3]] = []
            # print(file.split('/')[-3])
            self.noise_list[file.split('/')[-3]].append(file)
        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def add_reverberate(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float64), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))

        _, length = audio.shape
        return signal.convolve(audio, rir, mode='full')[:, :length]

    def add_noise(self, audio, noise_category):
        if self.musan_path is None:
            return audio
        _, length = audio.shape
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)
        num_noise = self.num_noise[noise_category]
        noise_list = random.sample(
            self.noise_list[noise_category], random.randint(num_noise[0], num_noise[1]))
        noises = []
        for noise in noise_list:
            noise_audio, sr = soundfile.read(noise)
            if noise_audio.shape[0] <= length:
                shortage = length - noise_audio.shape[0]
                noise_audio = np.pad(noise_audio, (0, shortage), 'wrap')
            start_frame = np.int64(
                random.random() * (noise_audio.shape[0] - length))
            noise_audio = noise_audio[start_frame:start_frame + length]
            noise_audio = np.stack([noise_audio], axis=0)
            noise_db = 10 * np.log10(np.mean(noise_audio ** 2) + 1e-4)
            noise_snr = random.uniform(
                self.noises_nr[noise_category][0], self.noises_nr[noise_category][1])
            noises.append(
                np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noise_audio)

        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        return audio + noise
