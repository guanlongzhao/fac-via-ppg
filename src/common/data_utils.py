# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA Corporation
# Copyright (c) 2019, Guanlong Zhao
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

import pickle
import random
import numpy as np
import torch
import torch.utils.data
from common.utils import load_filepaths
from common.utterance import Utterance
from common import layers
from ppg import DependenciesPPG
from scipy.io import wavfile
from common import feat
from common import ppg


# First order, dx(t) = 0.5(x(t + 1) - x(t - 1))
DELTA_WIN = [0, -0.5, 0.0, 0.5, 0]
# Second order, ddx(t) = 0.5(dx(t + 1) - dx(t - 1)) = 0.25(x(t + 2) - 2x(t)
# + x(t - 2))
ACC_WIN = [0.25, 0, -0.5, 0, 0.25]


def get_ppg(wav_path, deps):
    fs, wav = wavfile.read(wav_path)
    wave_data = feat.read_wav_kaldi_internal(wav, fs)
    seq = ppg.compute_full_ppg_wrapper(wave_data, deps.nnet, deps.lda, 10)
    return seq


def compute_dynamic_vector(vector, dynamic_win, frame_number):
    """Modified from https://github.com/CSTR-Edinburgh/merlin/blob/master
    /srcfrontend/acoustic_base.py
    Compute dynamic features for a data vector.
    Args:
        vector: A T-dim vector.
        dynamic_win: What type of dynamic features to compute. See DELTA_WIN
        and ACC_WIN.
        frame_number: The dimension of 'vector'.
    Returns:
        Dynamic feature vector.
    """
    vector = np.reshape(vector, (frame_number, 1))

    win_length = len(dynamic_win)
    win_width = int(win_length / 2)
    temp_vector = np.zeros((frame_number + 2 * win_width, 1))
    dynamic_vector = np.zeros((frame_number, 1))

    temp_vector[win_width:frame_number + win_width] = vector
    for w in range(win_width):
        temp_vector[w, 0] = vector[0, 0]
        temp_vector[frame_number + win_width + w, 0] = vector[
            frame_number - 1, 0]

    for i in range(frame_number):
        for w in range(win_length):
            dynamic_vector[i] += temp_vector[i + w, 0] * dynamic_win[w]

    return dynamic_vector


def compute_dynamic_matrix(data_matrix, dynamic_win):
    """Modified from https://github.com/CSTR-Edinburgh/merlin/blob/master
    /srcfrontend/acoustic_base.py
    Compute dynamic features for a data matrix. Calls compute_dynamic_vector
    for each feature dimension.
    Args:
        data_matrix: A (T, D) matrix.
        dynamic_win: What type of dynamic features to compute. See DELTA_WIN
        and ACC_WIN.
    Returns:
        Dynamic feature matrix.
    """
    frame_number, dimension = data_matrix.shape
    dynamic_matrix = np.zeros((frame_number, dimension))

    # Compute dynamic feature dimension by dimension
    for dim in range(dimension):
        dynamic_matrix[:, dim:dim + 1] = compute_dynamic_vector(
            data_matrix[:, dim], dynamic_win, frame_number)

    return dynamic_matrix


def compute_delta_acc_feat(matrix, is_delta=False, is_acc=False):
    """A wrapper to compute both the delta and delta-delta features and
    append them to the original features.
    Args:
        matrix: T*D matrix.
        is_delta: If set to True, compute delta features.
        is_acc: If set to True, compute delta-delta features.
    Returns:
        matrix: T*D (no dynamic feature) | T*2D (one dynamic feature) | T*3D
        (two dynamic features) matrix. Original feature matrix concatenated
    """
    if not is_delta and is_acc:
        raise ValueError('To use delta-delta feats you have to also use '
                         'delta feats.')
    if is_delta:
        delta_mat = compute_dynamic_matrix(matrix, DELTA_WIN)
    if is_acc:
        acc_mat = compute_dynamic_matrix(matrix, ACC_WIN)
    if is_delta:
        matrix = np.concatenate((matrix, delta_mat), axis=1)
    if is_acc:
        matrix = np.concatenate((matrix, acc_mat), axis=1)
    return matrix


def append_ppg(feats, f0):
    """Append log F0 and its delta and acc

    Args:
        feats:
        f0:

    Returns:

    """
    num_feats_frames = feats.shape[0]
    num_f0_frames = f0.shape[0]
    final_num_frames = min([num_feats_frames, num_f0_frames])
    feats = feats[:final_num_frames, :]
    f0 = f0[:final_num_frames]
    lf0 = np.log(f0 + np.finfo(float).eps)  # Log F0.
    lf0 = lf0.reshape(lf0.shape[0], 1)  # Convert to 2-dim matrix.
    lf0 = compute_delta_acc_feat(lf0, True, True)
    return np.concatenate((feats, lf0), axis=1)


class PPGMelLoader(torch.utils.data.Dataset):
    """Loads [ppg, mel] pairs."""

    def __init__(self, data_utterance_paths, hparams):
        """Data loader for the PPG->Mel task.

        Args:
            data_utterance_paths: A text file containing a list of file paths.
            hparams: The hyper-parameters.
        """
        self.data_utterance_paths = load_filepaths(data_utterance_paths)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.is_full_ppg = hparams.is_full_ppg
        self.is_append_f0 = hparams.is_append_f0
        self.is_cache_feats = hparams.is_cache_feats
        self.load_feats_from_disk = hparams.load_feats_from_disk
        self.feats_cache_path = hparams.feats_cache_path
        self.ppg_subsampling_factor = hparams.ppg_subsampling_factor
        self.ppg_deps = DependenciesPPG()

        if self.is_cache_feats and self.load_feats_from_disk:
            raise ValueError('If you are loading feats from the disk, do not '
                             'rewrite them back!')

        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_acoustic_feat_dims, hparams.sampling_rate,
            hparams.mel_fmin, hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.data_utterance_paths)

        self.ppg_sequences = []
        self.acoustic_sequences = []
        if self.load_feats_from_disk:
            print('Loading data from %s.' % self.feats_cache_path)
            with open(self.feats_cache_path, 'rb') as f:
                data = pickle.load(f)
            self.ppg_sequences = data[0]
            self.acoustic_sequences = data[1]
        else:
            for utterance_path in self.data_utterance_paths:
                ppg_feat_pair = self.extract_utterance_feats(utterance_path,
                                                   self.is_full_ppg)
                self.ppg_sequences.append(ppg_feat_pair[0].astype(
                    np.float32))
                self.acoustic_sequences.append(ppg_feat_pair[1])
        if self.is_cache_feats:
            print('Caching data to %s.' % self.feats_cache_path)
            with open(self.feats_cache_path, 'wb') as f:
                pickle.dump([self.ppg_sequences, self.acoustic_sequences], f)

    def extract_utterance_feats(self, data_utterance_path, is_full_ppg=False):
        """Get PPG and Mel (+ optional F0) for an utterance.

        Args:
            data_utterance_path: The path to the data utterance protocol buffer.
            is_full_ppg: If True, will use the full PPGs.

        Returns:
            feat_pairs: A list, each is a [pps, mel] pair.
        """
        utt = Utterance()
        fs, wav = wavfile.read(data_utterance_path)
        utt.fs = fs
        utt.wav = wav
        utt.ppg = get_ppg(data_utterance_path, self.ppg_deps)

        audio = torch.FloatTensor(utt.wav.astype(np.float32))
        fs = utt.fs

        if fs != self.stft.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                fs, self.stft.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        # (1, n_mel_channels, T)
        acoustic_feats = self.stft.mel_spectrogram(audio_norm)
        # (n_mel_channels, T)
        acoustic_feats = torch.squeeze(acoustic_feats, 0)
        # (T, n_mel_channels)
        acoustic_feats = acoustic_feats.transpose(0, 1)

        if is_full_ppg:
            if self.is_append_f0:
                ppg_f0 = append_ppg(utt.ppg, utt.f0)
                return [ppg_f0, acoustic_feats]
            else:
                return [utt.ppg, acoustic_feats]
        else:
            if self.is_append_f0:
                ppg_f0 = append_ppg(utt.monophone_ppg, utt.f0)
                return [ppg_f0, acoustic_feats]
            else:
                return [utt.monophone_ppg, acoustic_feats]

    def __getitem__(self, index):
        """Get a new data sample in torch.float32 format.

        Args:
            index: An int.

        Returns:
            T*D1 PPG sequence, T*D2 mels
        """
        if self.ppg_subsampling_factor == 1:
            curr_ppg = self.ppg_sequences[index]
        else:
            curr_ppg = self.ppg_sequences[index][
                       0::self.ppg_subsampling_factor, :]

        return torch.from_numpy(curr_ppg), self.acoustic_sequences[index]

    def __len__(self):
        return len(self.ppg_sequences)


def ppg_acoustics_collate(batch):
    """Zero-pad the PPG and acoustic sequences in a mini-batch.

    Also creates the stop token mini-batch.

    Args:
        batch: An array with B elements, each is a tuple (PPG, acoustic).
        Consider this is the return value of [val for val in dataset], where
        dataset is an instance of PPGSpeechLoader.

    Returns:
        ppg_padded: A (batch_size, feature_dim_1, num_frames_1) tensor.
        input_lengths: A batch_size array, each containing the actual length
        of the input sequence.
        acoustic_padded: A (batch_size, feature_dim_2, num_frames_2) tensor.
        gate_padded: A (batch_size, num_frames_2) tensor. If "1" means reaching
        stop token. Currently assign "1" at the last frame and the padding.
        output_lengths: A batch_size array, each containing the actual length
        of the output sequence.
    """
    # Right zero-pad all PPG sequences to max input length.
    # x is (PPG, acoustic), x[0] is PPG, which is an (L(varied), D) tensor.
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([x[0].shape[0] for x in batch]), dim=0,
        descending=True)
    max_input_len = input_lengths[0]
    ppg_dim = batch[0][0].shape[1]

    ppg_padded = torch.FloatTensor(len(batch), max_input_len, ppg_dim)
    ppg_padded.zero_()
    for i in range(len(ids_sorted_decreasing)):
        curr_ppg = batch[ids_sorted_decreasing[i]][0]
        ppg_padded[i, :curr_ppg.shape[0], :] = curr_ppg

    # Right zero-pad acoustic features.
    feat_dim = batch[0][1].shape[1]
    max_target_len = max([x[1].shape[0] for x in batch])
    # Create acoustic padded and gate padded
    acoustic_padded = torch.FloatTensor(len(batch), max_target_len, feat_dim)
    acoustic_padded.zero_()
    gate_padded = torch.FloatTensor(len(batch), max_target_len)
    gate_padded.zero_()
    output_lengths = torch.LongTensor(len(batch))
    for i in range(len(ids_sorted_decreasing)):
        curr_acoustic = batch[ids_sorted_decreasing[i]][1]
        acoustic_padded[i, :curr_acoustic.shape[0], :] = curr_acoustic
        gate_padded[i, curr_acoustic.shape[0] - 1:] = 1
        output_lengths[i] = curr_acoustic.shape[0]

    ppg_padded = ppg_padded.transpose(1, 2)
    acoustic_padded = acoustic_padded.transpose(1, 2)

    return ppg_padded, input_lengths, acoustic_padded, gate_padded,\
        output_lengths


def utt_to_sequence(utt: Utterance, is_full_ppg=False, is_append_f0=False):
    """Get PPG tensor for inference.

    Args:
        utt: A data utterance object.
        is_full_ppg: If True, will use the full PPGs.
        is_append_f0: If True, will append F0 features.

    Returns:
        A 1*D*T tensor.
    """
    if is_full_ppg:
        ppg = utt.ppg
    else:
        ppg = utt.monophone_ppg

    if is_append_f0:
        ppg = append_ppg(ppg, utt.f0)

    return torch.from_numpy(ppg).float().transpose(0, 1).unsqueeze(0)
