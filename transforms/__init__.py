from .transform_wav import *
from torchvision.transforms import Compose
from datasets import AugmentationDataset


def build_transform(audio_config,
                    mode,
                    num_stack=1,
                    noise_path=None):
    duration = audio_config['duration']
    add_sample = audio_config['add_sample']
    transform = FixAudioLength(time=duration,
                               add_sample=add_sample,
                               num=num_stack)
    if mode == 'train':
        if noise_path is not None:
            bg_dataset = AugmentationDataset(musan_path=noise_path['musan_path'],
                                             rir_path=noise_path['rir_path'])

            augment = Augmentation(bg_dataset=bg_dataset)
        return Compose([transform, augment, ToTensor('samples', 'input')])
    return Compose([transform, ToTensor('samples', 'input')])
