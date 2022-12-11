import numpy as np
import soundfile
import torch
import random
import librosa
import utils


class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1, add_sample=0, num=1):
        self.time = time
        self.add_sample = add_sample
        self.num = num

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        max_length = int(self.time * sample_rate) + self.add_sample
        if len(samples) <= max_length:
            shortage = max_length - len(samples)
            samples = np.pad(samples, (0, shortage), "constant")
        start_frame = np.int64(
                random.random() * (samples.shape[0] - max_length))
        samples = samples[start_frame:start_frame + max_length]
        if self.num != 1:
            feats = []
            start_frame = np.linspace(0, len(samples) - max_length, num=self.num)
            for asf in start_frame:
                feats.append(samples[int(asf):int(asf) + max_length])
            feats = np.stack(feats, axis=0).astype(np.float32)
            data['samples'] = feats
        else:
            data['samples'] = np.stack([samples], axis=0)
        return data


class ChangeAmplitude(object):
    """Changes amplitude of an audio randomly."""

    def __init__(self, amplitude_range=(0.7, 1.1), prob=0.5):
        self.amplitude_range = amplitude_range
        self.prob = prob

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        data['samples'] = data['samples'] * random.uniform(*self.amplitude_range)
        return data


class ChangeSpeedAndPitchAudio(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""

    def __init__(self, max_scale=0.2, prob=0.5):
        self.max_scale = max_scale
        self.prob = prob

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        samples = data['samples']
        scale = random.uniform(-self.max_scale, self.max_scale)
        speed_fac = 1.0 / (1 + scale)
        data['samples'] = np.interp(np.arange(0, len(samples), speed_fac), np.arange(0, len(samples)), samples).astype(
            np.float64)
        return data


class StretchAudio(object):
    """Stretches an audio randomly."""

    def __init__(self, max_scale=0.2, prob=0.5):
        self.max_scale = max_scale
        self.prob = prob

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        scale = random.uniform(-self.max_scale, self.max_scale)
        data['samples'] = librosa.effects.time_stretch(y=data['samples'], rate=1 + scale)
        return data


class ShiftAudio(object):
    """Shift audio randomly """

    def __init__(self, min_shift, max_shift, prob=0.5):
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.prob = prob

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data
        sample_rate = data['sample_rate']
        shift_range = int(
            np.random.uniform(round(self.min_shift * sample_rate / 1000), round(self.max_shift * sample_rate / 1000)))
        # print(shift_range)
        data['samples'] = np.roll(data['samples'], shift_range)
        return data


class ToTensor(object):
    """Converts into a tensor."""

    def __init__(self, np_name, tensor_name, normalize=None,mode='train',stage=1):
        self.np_name = np_name
        self.tensor_name = tensor_name
        self.normalize = normalize
        self.mode = mode
        self.stage = stage
    def __call__(self, data):
        d = data[self.np_name]
        # Check stack when eval
        if self.stage == 1:
            tensor = torch.FloatTensor(d[0])
        else:
            if self.mode == 'train':
                tensor = torch.FloatTensor(d[0])
            else:
                tensor = torch.FloatTensor(d)
        # tensor = torch.FloatTensor(data[self.np_name])
        if self.normalize is not None:
            mean, std = self.normalize
            tensor -= mean
            tensor /= std
        data[self.tensor_name] = tensor
        return data


class Augmentation(object):
    """ Add background noise to wave"""

    def __init__(self, bg_dataset):
        self.bg_dataset = bg_dataset

    def __call__(self, data):
        audio = data['samples']
        augtype = random.randint(0, 5)
        # audio = self.bg_dataset.add_reverberate(audio)

        if augtype == 0:  # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = self.bg_dataset.add_reverberate(audio)
        elif augtype == 2:  # Babble
            audio = self.bg_dataset.add_noise(audio, 'speech')
        elif augtype == 3:  # Music
            audio = self.bg_dataset.add_noise(audio, 'music')
        elif augtype == 4:  # Noise
            audio = self.bg_dataset.add_noise(audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = self.bg_dataset.add_noise(audio, 'speech')
            audio = self.bg_dataset.add_noise(audio, 'music')
        data['samples'] = audio
        return data


# debug
if __name__ == '__main__':
    from torchvision.transforms import Compose
    from datasets import AugmentationDataset

    noise_dataset = AugmentationDataset(rir_path='F:\\Datasets\\keyword-spotting\\RIRS_NOISES\\simulated_rirs',
                                        musan_path='F:/Datasets/keyword-spotting/musan/musan')
    trans = Compose([FixAudioLength(time=2, add_sample=240, num=1),
                    #  Augmentation(bg_dataset=noise_dataset),
                    #  ToTensor('samples', 'input')
                     ],
                    )
    data = {}
    data['samples'] = utils.load_audio('../utils/temp.wav', 16000)[0]
    # print(data['samples'].shape)
    data['sample_rate'] = 16000
    data = trans(data)
    from IPython import embed
    embed()
    # data['input'].size())
    # print(data['samples'][0].shape)
    # # torchaudio.save('sound.wav', data['samples'], 16000)
    # soundfile.write('sample.wav',data['samples'][0],16000)
