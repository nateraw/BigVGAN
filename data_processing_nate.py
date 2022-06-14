#https://console.cloud.google.com/compute/instancesDetail/zones/us-central1-a/instances/nate-gpu-3?project=huggingface-ml&pageState=(%22duration%22:(%22groupValue%22:%22PT1H%22,%22customValue%22:null))
from mel_processing import spectrogram_torch
import torch
import librosa
from scipy.io.wavfile import read
import numpy as np
from utils import HParams
import json
import scipy.signal as sps

def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def downsample(audio, sr):
    num = round(len(audio)*float(22050) / float(sr))
    return sps.resample(audio, num)

def process(audio_filepath, hparams):
    audio, sr = load_wav_to_torch(audio_filepath)
    audio, _ = librosa.effects.trim(
        np.array(audio), 
        top_db=hparams.data.top_db,
        frame_length=hparams.data.filter_length,
        hop_length=hparams.data.hop_length
    )

    if sr != hparams.data.sampling_rate:
        audio = downsample(audio, sr)

    audio = torch.FloatTensor(audio.astype(np.float32))
    audio_norm = audio / hparams.data.max_wav_value  * 0.95
    audio_norm = audio_norm.unsqueeze(0)

    spec = spectrogram_torch(
        audio_norm,
        hparams.data.filter_length,
        hparams.data.sampling_rate,
        hparams.data.hop_length,
        hparams.data.win_length,
        center=False
    )
    spec = torch.squeeze(spec, 0)
    return spec, audio_norm


def collate_fn(batch):

    max_spec_len = max([x[0].size(1) for x in batch])
    max_wav_len = max([x[1].size(1) for x in batch])

    spec_lengths = torch.LongTensor(len(batch))
    wav_lengths = torch.LongTensor(len(batch))

    spec_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_spec_len)
    wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

    spec_padded.zero_()
    wav_padded.zero_()

    for i in range(len(batch)):
        row = batch[i]

        spec = row[0]
        spec_padded[i, :, :spec.size(1)] = spec
        spec_lengths[i] = spec.size(1)

        wav = row[1]
        wav_padded[i, :, :wav.size(1)] = wav
        wav_lengths[i] = wav.size(1)

    return spec_padded, spec_lengths, wav_padded, wav_lengths

with open('configs/vctk_bigvgan.json', 'r') as f:
    data = f.read()
hparams = HParams(**json.loads(data))

paths = [
    './AudioMNIST/data/01/0_01_0.wav',
    './AudioMNIST/data/02/0_02_0.wav',
    './AudioMNIST/data/03/0_03_0.wav',
    './AudioMNIST/data/04/0_04_0.wav',
    './AudioMNIST/data/05/0_05_0.wav',
    './AudioMNIST/data/06/0_06_0.wav',
    './AudioMNIST/data/07/0_07_0.wav',
    './AudioMNIST/data/08/0_08_0.wav',
]
batch = []
for path in paths:
    spec, audio_norm = process(path, hparams)
    # if spec.size(-1) >= 32:
    batch.append([spec, audio_norm])

spec, spec_lengths, y, y_lengths = collate_fn(batch)
# out = process('audio/target1_30k_step.wav', hparams)


################
# Testing
################

# from models_bigvgan import SynthesizerTrn, MultiPeriodDiscriminator
# import commons
# from losses import generator_loss, discriminator_loss, feature_loss
# from torch.nn import functional as F
# from mel_processing import mel_spectrogram_torch, spec_to_mel_torch


# hparams.model.batch_size = len(batch)
# generator = SynthesizerTrn(
#     hparams.data.filter_length // 2 + 1,
#     hparams.train.segment_size // hparams.data.hop_length,
#     **vars(hparams.model),
# ).cuda()
# discriminator = MultiPeriodDiscriminator(hparams.model.use_spectral_norm).cuda()
# optim_g = torch.optim.AdamW(
#     generator.parameters(),
#     hparams.train.learning_rate,
#     betas=hparams.train.betas,
#     eps=hparams.train.eps
# )
# optim_d = torch.optim.AdamW(
#     discriminator.parameters(),
#     hparams.train.learning_rate,
#     betas=hparams.train.betas,
#     eps=hparams.train.eps
# )

# # First set of losses
# y_hat, ids_slice = generator(spec.cuda(), spec_lengths.cuda())
# mel = spec_to_mel_torch(
#     spec.float(),
#     hparams.data.filter_length,
#     hparams.data.n_mel_channels,
#     hparams.data.sampling_rate,
#     hparams.data.mel_fmin,
#     hparams.data.mel_fmax
# )
# y_hat_mel = mel_spectrogram_torch(
#     y_hat.float().squeeze(1),
#     hparams.data.filter_length,
#     hparams.data.n_mel_channels,
#     hparams.data.sampling_rate,
#     hparams.data.hop_length,
#     hparams.data.win_length,
#     hparams.data.mel_fmin,
#     hparams.data.mel_fmax
# )

# y_mel = commons.slice_segments(mel, ids_slice, hparams.train.segment_size // hparams.data.hop_length)
# y = commons.slice_segments(y, ids_slice * hparams.data.hop_length, hparams.train.segment_size).cuda()
# y_d_hat_r, y_d_hat_g, _, _ = discriminator(y, y_hat.detach())
# loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g) # Real vs generated
# loss_disc_all = loss_disc

# optim_d.zero_grad()
# loss_disc_all.backward()
# optim_d.step()


# y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = discriminator(y, y_hat)
# loss_mel = F.l1_loss(y_mel.cuda(), y_hat_mel) * hparams.train.c_mel
# loss_fm = feature_loss(fmap_r, fmap_g)
# loss_gen, losses_gen = generator_loss(y_d_hat_g)
# loss_gen_all = loss_gen + loss_fm + loss_mel

# optim_g.zero_grad()
# loss_gen_all.backward()
# optim_g.step()
