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
                 gender='mix',
                 aug_path=None,
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
        df = pd.read_csv(path_to_df)
        if stage == 1 and gender != 'mix':
            df = df[df['gender'] == gender]
        self.df = df
        self.classes = classes
        self.n_class = len(classes)
        self.sample_rate = sample_rate
        self.transform = transform
        self.dataset_name = dataset_name
        self.stage = stage
        self.bg_dataset = None
        if aug_path is not None:
            self.bg_dataset = AugmentationDataset(musan_path=aug_path['musan_path'],
                                             rir_path=aug_path['rir_path'])

            # augment = Augmentation(bg_dataset=bg_dataset)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row_data = self.df.iloc[idx]
        wav, _ = utils.load_audio(os.path.join(
            self.root_dir, str(row_data['file'])), self.sample_rate)
        # print(row_data)
        label = str(row_data['speaker'])
        if self.dataset_name == 'audio_mnist':
            if int(row_data['speaker']) < 10:
                label = '0' + label
        target = utils.label2index(self.classes,label)
        data = {
            'samples': wav,
            'sample_rate': self.sample_rate,
            'target': target,
            'path': row_data['file']
        }

        if self.stage == 1:
            data['word'] = str(row_data['word']),
        # if self.bg_dataset is not None:
        #     augtype = random.randint(0, 5)
        # # audio = self.bg_dataset.add_reverberate(audio)

        #     if augtype == 0:  # Original
        #         audio = audio
        #     elif augtype == 1:  # Reverberation
        #         audio = self.bg_dataset.add_reverberate(audio)
        #     elif augtype == 2:  # Babble
        #         audio = self.bg_dataset.add_noise(audio, 'speech')
        #     elif augtype == 3:  # Music
        #         audio = self.bg_dataset.add_noise(audio, 'music')
        #     elif augtype == 4:  # Noise
        #         audio = self.bg_dataset.add_noise(audio, 'noise')
        #     elif augtype == 5:  # Television noise
        #         audio = self.bg_dataset.add_noise(audio, 'speech')
        #         audio = self.bg_dataset.add_noise(audio, 'music')

        if self.transform is not None:
            data = self.transform(data)

        return data


class BackgroundNoiseDataset(Dataset):
    def __init__(self, paths,sample_rate, sample_length=1,add_sample=240):

        noise_files = []
        for path in paths:
            noise_files.extend([os.path.join(path,file) for file in os.listdir(path) if file.endswith('.wav')])


        samples = []
        for f in noise_files:
            sample, sample_rate = utils.load_audio(f, sample_rate)
            samples.append(sample)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length) + add_sample
        r = len(samples) // c
        self.samples = samples[:r * c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.paths = paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {
            'samples': self.samples[index],
            'sample_rate': self.sample_rate,
            'target': 1,
            'paths': self.paths
        }
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


class VoxCelebDataset(Dataset):
    def __init__(self, train_list, train_path,sample_rate=16000, transform=None):
        super(VoxCelebDataset, self).__init__()
        self.train_path = train_path
        self.transform = transform
        self.sample_rate = sample_rate
        # Load data & labels
        self.data_list  = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name     = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        wav, _ = utils.load_audio(
            self.data_list[idx], self.sample_rate)
        # print(row_data)
        # label = str(row_data['speaker'])
        # if self.dataset_name == 'audio_mnist':
        #     if int(row_data['speaker']) < 10:
        #         label = '0' + label
        # target = utils.label2index(self.classes,label)
        data = {
            'samples': wav,
            'sample_rate': self.sample_rate,
            'target': self.data_label[idx],
            # 'path': self.data_list[idx]
        }

        # if self.stage == 1:
        #     data['word'] = str(row_data['word']),

        if self.transform is not None:
            data = self.transform(data)

        return data
