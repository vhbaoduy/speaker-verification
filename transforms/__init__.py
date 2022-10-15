from .transform_wav import *
from torchvision.transforms import Compose
from datasets import BackgroundNoiseDataset


def build_transform(audio_config,
                    mode,
                    noise_path=None):
    duration = audio_config['duration']
    if mode == 'train':
        if audio_config['augment']:
            augmentor = audio_config['Augmentor']
            prob = augmentor['prob']
            noise_prob = augmentor['noise_prob']
            min_shift, max_shift = augmentor['min_shift'], augmentor['max_shift'
            ]
            data_transform = Compose(
                [ChangeAmplitude(prob=prob),
                 ShiftAudio(min_shift=min_shift, max_shift=max_shift, prob=prob),
                 StretchAudio(prob=prob),
                 ChangeSpeedAndPitchAudio(prob=prob),
                 FixAudioLength(time=duration),
                 ])
        else:
            data_transform = FixAudioLength()

        if noise_path is not None:
            bg_dataset = BackgroundNoiseDataset(path=noise_path,
                                                transform=data_transform,
                                                sample_rate=audio_config['sample_rate'])
            transform = Compose([data_transform,
                                 AddBackgroundNoise(bg_dataset=bg_dataset,
                                                    prob=noise_prob)])
        else:
            transform = data_transform
    else:
        transform = FixAudioLength(time=duration)

    return Compose([transform, ToTensor('samples', 'input')])
