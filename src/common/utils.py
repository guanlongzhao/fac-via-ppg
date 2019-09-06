# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Modified from https://github.com/NVIDIA/tacotron2"""

import numpy as np
from scipy.io.wavfile import read
import torch
from scipy import signal


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def get_mask_from_lengths_window_and_time_step(lengths, attention_window_size,
                                               time_step):
    """
    One for mask and 0 for not mask
    Args:
        lengths:
        attention_window_size:
        time_step: zero-indexed

    Returns:

    """
    # Mask all initially.
    max_len = torch.max(lengths).item()
    B = len(lengths)
    mask = torch.cuda.ByteTensor(B, max_len)
    mask[:] = 1

    for ii in range(B):
        # Note that the current code actually have a minor side effect,
        # where the utterances that are shorter than the longest one will
        # still have their actual last time step unmasked when the decoding
        # passes beyond that time step. I keep this bug here simply because
        # it will prevent numeric errors when computing the attention weights.
        max_idx = lengths[ii] - 1
        # >=0, <= the actual sequence end idx (length-1) (not covered here)
        start_idx = min([max([0, time_step-attention_window_size]), max_idx])
        # <=length-1
        end_idx = min([time_step+attention_window_size, max_idx])
        if start_idx > end_idx:
            continue
        mask[ii, start_idx:(end_idx+1)] = 0
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [tuple(line.strip().split(split)) for line in f]
    return filepaths_and_text


def load_filepaths(filename):
    """Read in a list of file paths.

    Args:
        filename: A text file containing a list of file paths. Assume that
        each line has one file path.

    Returns:
        filepaths: A list of strings where each is a file path.
    """
    with open(filename) as f:
        filepaths = [line.strip() for line in f]
    return filepaths


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


def notch_filtering(wav, fs, w0, Q):
    """ Apply a notch (band-stop) filter to the audio signal.

    Args:
        wav: Waveform.
        fs: Sampling frequency of the waveform.
        w0: See scipy.signal.iirnotch.
        Q: See scipy.signal.iirnotch.

    Returns:
        wav: Filtered waveform.
    """
    b, a = signal.iirnotch(2 * w0/fs, Q)
    wav = signal.lfilter(b, a, wav)
    return wav


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