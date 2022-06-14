import utils
import argparse
import json
import glob
import os
import numpy as np
import librosa
from tqdm import tqdm
import scipy.signal as sps
from text import text_to_sequence
import torch
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/vctk_bigvgan.json",
                      help='JSON file for configuration')
    parser.add_argument('-i', '--input_path', type=str, default="./data")
    parser.add_argument('-o', '--output_path', type=str, default="./data/preprocessed_npz")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = utils.HParams(**config)

    speakers = [os.path.basename(i) for i in glob.glob(os.path.join(args.input_path,'wav48_silence_trimmed/*'))]

    for speaker in tqdm(speakers):
        os.makedirs(os.path.join(args.output_path,speaker,'train'),exist_ok=True)
        os.makedirs(os.path.join(args.output_path,speaker,'test'),exist_ok=True)

        wavs = sorted(glob.glob(os.path.join(args.input_path,'wav48_silence_trimmed',speaker,'*.flac')))
        # print("WAVS", wavs)
        for wav in wavs[:25]:
            data = preprocess_wav(wav, hparams)
            np.savez(os.path.join(args.output_path,speaker,'test',os.path.basename(wav).replace('.flac','.npz')),
                    **data, allow_pickle=False)
        
        for wav in wavs[25:]:
            data = preprocess_wav(wav, hparams)
            np.savez(os.path.join(args.output_path,speaker,'train',os.path.basename(wav).replace('.flac','.npz')),
                    **data, allow_pickle=False)

def downsample(audio, sr):
    num = round(len(audio)*float(22050) / float(sr))
    return sps.resample(audio, num)

def preprocess_wav(wav, hparams):
    # audio, sr = utils.load_wav_to_torch(wav)

    audio, sr = librosa.load(wav)
    audio = torch.FloatTensor(audio.astype(np.float32))
    audio, _ = librosa.effects.trim(np.array(audio), 
                                    top_db=hparams.data.top_db,
                                    frame_length=hparams.data.filter_length,
                                    hop_length=hparams.data.hop_length)

    if sr != hparams.data.sampling_rate:
        audio = downsample(audio, sr)
    
    text_file = wav.replace('wav48_silence_trimmed','txt').replace('.flac','.txt')
    text_file = re.sub('_mic\d', '', text_file)
    with open(text_file, encoding='utf8') as f:
        text = f.readline().rstrip()
    token = text_to_sequence(text, ["english_cleaners2"]) 

    data = {
        'audio': audio,
        'token': token,
        'text': text
    }

    return data




if __name__ == "__main__":
    main()
