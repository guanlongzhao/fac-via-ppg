# Copyright 2019 Guanlong Zhao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from common.hparams import create_hparams_stage
from script.train_ppg2mel import load_model
from common.utils import to_gpu
from common.layers import TacotronSTFT
from common import feat
from scipy.io import wavfile
import numpy as np
import sys
import torch
import ppg
import os
import logging
import datetime
import time
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
#                              'src', 'waveglow'))
from waveglow.denoiser import Denoiser
from common.data_utils import get_ppg


def get_mel(wav, stft):
    audio = torch.FloatTensor(wav.astype(np.float32))
    audio_norm = audio / 32768
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    # (1, n_mel_channels, T)
    acoustic_feats = stft.mel_spectrogram(audio_norm)
    return acoustic_feats


def waveglow_audio(mel, waveglow, sigma, is_cuda_output=False):
    mel = torch.autograd.Variable(mel.cuda())
    if not is_cuda_output:
        with torch.no_grad():
            audio = 32768 * waveglow.infer(mel, sigma=sigma)[0]
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
    else:
        with torch.no_grad():
            audio = waveglow.infer(mel, sigma=sigma).cuda()
    return audio


def get_inference(seq, model, is_clip=False):
    """Tacotron inference.

    Args:
        seq: T*D numpy array.
        model: Tacotron model.
        is_clip: Set to True to avoid the artifacts at the end.

    Returns:
        synthesized mels.
    """
    # (T, D) numpy -> (1, D, T) cpu tensor
    seq = torch.from_numpy(seq).float().transpose(0, 1).unsqueeze(0)
    # cpu tensor -> gpu tensor
    seq = to_gpu(seq)
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(seq)
    if is_clip:
        return mel_outputs_postnet[:, :, 10:(seq.size(2)-10)]
    else:
        return mel_outputs_postnet


def load_waveglow_model(path):
    model = torch.load(path)['model']
    model = model.remove_weightnorm(model)
    model.cuda().eval()
    return model


if __name__ == '__main__':
    # Prepare dirs
    timestamp = datetime.datetime.fromtimestamp(time.time())
    output_dir = \
        '/media/guanlong/DATA1/exp/ppg-speech/samples/trial_%04d%02d%02d' \
        '-%02d%02d%02d' \
        % (timestamp.year, timestamp.month, timestamp.day, timestamp.hour,
           timestamp.minute, timestamp.second)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    logging.basicConfig(filename=os.path.join(output_dir, 'debug.log'),
                        level=logging.DEBUG)
    logging.info('Output dir: %s', output_dir)

    # Parameters
    checkpoint_path = ''
    teacher_utt_path = ''
    waveglow_path = ''
    is_clip = False  # Set to True to control the output length of AC.
    fs = 16000
    waveglow_sigma = 0.6
    waveglow_for_denoiser = torch.load(waveglow_path)['model']
    waveglow_for_denoiser.cuda()
    denoiser_mode = 'zeros'
    denoiser = Denoiser(waveglow_for_denoiser, mode=denoiser_mode)
    denoiser_strength = 0.005
    # End of parameters

    logging.debug('Tacotron: %s', checkpoint_path)
    logging.debug('Waveglow: %s', waveglow_path)
    logging.debug('AM: SI model')
    logging.debug('is_clip: %d', is_clip)
    logging.debug('Fs: %d', fs)
    logging.debug('Sigma: %f', waveglow_sigma)
    logging.debug('Denoiser strength: %f', denoiser_strength)
    logging.debug('Denoiser mode: %s', denoiser_mode)

    hparams = create_hparams_stage()
    taco_stft = TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_acoustic_feat_dims, hparams.sampling_rate,
        hparams.mel_fmin, hparams.mel_fmax)

    # Load models.
    tacotron_model = load_model(hparams)
    tacotron_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = tacotron_model.eval()
    waveglow_model = load_waveglow_model(waveglow_path)

    deps = ppg.DependenciesPPG()

    if os.path.isfile(teacher_utt_path):
        logging.info('Perform AC on %s', teacher_utt_path)
        teacher_ppg = get_ppg(teacher_utt_path, deps)
        ac_mel = get_inference(teacher_ppg, tacotron_model, is_clip)
        ac_wav = waveglow_audio(ac_mel, waveglow_model,
                                waveglow_sigma, True)
        ac_wav = denoiser(
            ac_wav, strength=denoiser_strength)[:, 0].cpu().numpy().T

        output_file = os.path.join(output_dir, 'ac.wav')
        wavfile.write(output_file, fs, ac_wav)
    else:
        logging.warning('Missing %s', teacher_utt_path)

    logging.info('Done!')
