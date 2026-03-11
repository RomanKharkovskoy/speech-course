import logging
import os

import torch
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS

logger = logging.getLogger(__name__)

TARGET_LENGTH = 16000


def pad_or_trim(waveform, target_length=TARGET_LENGTH):
    current_length = waveform.shape[1]
    if current_length < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - current_length))
    else:
        waveform = waveform[:, :target_length]
    return waveform


class AudioDataset(Dataset):
    def __init__(self, root="../data", subset="training"):
        self.dataset = SPEECHCOMMANDS(root=root, download=True, subset=subset)

        self.indices = []
        yes_count = 0
        no_count = 0

        for i in range(len(self.dataset)):
            filepath = self.dataset._walker[i]
            label = os.path.basename(os.path.dirname(filepath))
            if label in ("yes", "no"):
                self.indices.append(i)

                if label == "yes":
                    yes_count += 1
                else:
                    no_count += 1

        logger.info(f"{subset}: {len(self.indices)} samples (yes={yes_count}, no={no_count})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        waveform, sr, label, *_ = self.dataset[self.indices[idx]]
        waveform = pad_or_trim(waveform)
        target = 1.0 if label == "yes" else 0.0
        return waveform, target
