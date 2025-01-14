import pytorch_lightning as pl
from torch.nn import functional as F
import torch
import json
from torch.utils.data import DataLoader


from data_processing_nate import process, collate_fn
from process_audiomnist import AudioMNISTDataset, PathsDataset
from utils import HParams
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models_bigvgan import SynthesizerTrn, MultiPeriodDiscriminator
import commons
from losses import generator_loss, discriminator_loss, feature_loss
from data_utils_vocoder import (
    TextAudioLoader,
    TextAudioCollate,
    DistributedBucketSampler
)

class LitBigVGAN(pl.LightningModule):
    def __init__(self, train, data, model):
        super().__init__()
        self.save_hyperparameters()
        self.generator = SynthesizerTrn(
            self.hparams.data.filter_length // 2 + 1,
            self.hparams.train.segment_size // self.hparams.data.hop_length,
            **vars(self.hparams.model),
        )
        self.discriminator = MultiPeriodDiscriminator(hparams.model.use_spectral_norm)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # spec, spec_lengths, y, y_lengths = batch
        x, x_lengths, spec, spec_lengths, y, y_lengths, sid = batch
        y_hat, ids_slice = self.generator(spec.cuda(), spec_lengths.cuda())
        mel = spec_to_mel_torch(
            spec.float(),
            self.hparams.data.filter_length,
            self.hparams.data.n_mel_channels,
            self.hparams.data.sampling_rate,
            self.hparams.data.mel_fmin,
            self.hparams.data.mel_fmax
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.float().squeeze(1),
            self.hparams.data.filter_length,
            self.hparams.data.n_mel_channels,
            self.hparams.data.sampling_rate,
            self.hparams.data.hop_length,
            self.hparams.data.win_length,
            self.hparams.data.mel_fmin,
            self.hparams.data.mel_fmax
        )
        y_mel = commons.slice_segments(mel, ids_slice, self.hparams.train.segment_size // self.hparams.data.hop_length)
        y = commons.slice_segments(y, ids_slice * self.hparams.data.hop_length, self.hparams.train.segment_size).cuda()

        if optimizer_idx == 0:
            y_d_hat_r, y_d_hat_g, _, _ = self.discriminator(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
            loss_disc_all = loss_disc
            self.log("d_loss", loss_disc_all, prog_bar=True)
            return loss_disc_all

        if optimizer_idx == 1:
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(y, y_hat)
            loss_mel = F.l1_loss(y_mel.cuda(), y_hat_mel) * self.hparams.train.c_mel
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel
            self.log("g_loss", loss_gen_all, prog_bar=True)
            return loss_gen_all
    
    def configure_optimizers(self):
        optim_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            self.hparams.train.learning_rate,
            betas=self.hparams.train.betas,
            eps=self.hparams.train.eps
        )
        optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            self.hparams.train.learning_rate,
            betas=self.hparams.train.betas,
            eps=self.hparams.train.eps
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=self.hparams.train.lr_decay, last_epoch=-1)
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=self.hparams.train.lr_decay, last_epoch=-1)

        return [optim_d, optim_g], [{'scheduler': scheduler_d, 'interval': 'epoch'}, {'scheduler': scheduler_g, 'interval': 'epoch'}]

    def configure_gradient_clipping(self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm):
        if optimizer_idx == 0:
            commons.clip_grad_value_(self.discriminator.parameters(), None)
        if optimizer_idx == 1:
            commons.clip_grad_value_(self.generator.parameters(), None)

if __name__ == '__main__':
    with open('configs/vctk.json', 'r') as f:
        data = f.read()
    hparams = HParams(**json.loads(data))

    # ds = AudioMNISTDataset('./AudioMNIST/data', hparams)

    # ds = PathsDataset('paths_valid.txt', hparams)
    # train_loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)

    train_dataset = TextAudioLoader(hparams.data.training_files, hparams.data, is_train=True)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hparams.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=1,
        rank=0,
        shuffle=True
    )
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler
    )

    model = LitBigVGAN(hparams.train, hparams.data, hparams.model)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        log_every_n_steps=10,
        accumulate_grad_batches=8
    )

    trainer.fit(model, train_loader)