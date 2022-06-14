from data_processing_nate import load_wav_to_torch, downsample, process, collate_fn

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from utils import HParams
import json
import librosa
import numpy as np


class AudioMNISTDataset(Dataset):
    def __init__(self, root, hparams):
        self.root = Path(root)
        self.hparams = hparams
        assert self.root.exists()
        paths_all = list(self.root.glob('**/*.wav'))
        self.paths = []
        for path in paths_all:
            audio, sr = load_wav_to_torch(path)
            audio, _ = librosa.effects.trim(
                np.array(audio), 
                top_db=self.hparams.data.top_db,
                frame_length=self.hparams.data.filter_length,
                hop_length=self.hparams.data.hop_length
            )

            if sr != self.hparams.data.sampling_rate:
                audio = downsample(audio, sr)
            
            if audio.shape[-1] >= self.hparams.train.segment_size:
                self.paths.append(path)

    def __getitem__(self, idx):
        fpath = self.paths[idx]
        spec, audio_norm = process(fpath, self.hparams)
        return spec, audio_norm

    def __len__(self):
        return len(self.paths)

class PathsDataset(AudioMNISTDataset):
    def __init__(self, paths_file, hparams):
        self.paths = [x.strip() for x in Path(paths_file).read_text().split('\n')]
        self.hparams = hparams

# def process(fpath, hparams):
#     audio, sr = load_wav_to_torch(fpath)
#     audio, _ = librosa.effects.trim(
#         np.array(audio), 
#         top_db=hparams.data.top_db,
#         frame_length=hparams.data.filter_length,
#         hop_length=hparams.data.hop_length
#     )
#     if sr != hparams.data.sampling_rate:
#         audio = downsample(audio, sr)
#     print(audio.shape)

with open('configs/vctk_bigvgan.json', 'r') as f:
    data = f.read()
hparams = HParams(**json.loads(data))


# root = Path('./AudioMNIST/data')
# paths = root.glob('**/*.wav')

# for path in list(paths)[:10]:
#     process(path, hparams)

ds = AudioMNISTDataset('./AudioMNIST/data', hparams)
loader = DataLoader(ds, batch_size=32, collate_fn=collate_fn)