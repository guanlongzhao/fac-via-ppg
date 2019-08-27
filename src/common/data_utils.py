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

import logging
import os
import pickle
import random
import numpy as np
import torch
import torch.utils.data
from common.utils import load_filepaths, load_filepaths_and_text
from common.utterance import Utterance, is_sil
from common import layers
from ppg import DependenciesPPG, reduce_ppg_dim
from kaldi.matrix import Matrix
from scipy.io import wavfile
from common import feat
from common import ppg


# First order, dx(t) = 0.5(x(t + 1) - x(t - 1))
DELTA_WIN = [0, -0.5, 0.0, 0.5, 0]
# Second order, ddx(t) = 0.5(dx(t + 1) - dx(t - 1)) = 0.25(x(t + 2) - 2x(t)
# + x(t - 2))
ACC_WIN = [0.25, 0, -0.5, 0, 0.25]


def get_ppg(wav_path, deps, is_fmllr=False, speaker_code=None):
    fs, wav = wavfile.read(wav_path)
    wave_data = feat.read_wav_kaldi_internal(wav, fs)
    if is_fmllr:
        seq = ppg.compute_full_ppg_wrapper(wave_data, deps.nnet, deps.lda, 10,
                                           fmllr=deps.fmllr_mats[speaker_code])
    else:
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


class MeanVarianceNorm(object):
    """Adapted from https://github.com/CSTR-Edinburgh/merlin/blob/master/src
    /frontend/mean_variance_norm.py"""
    def __init__(self, acoustic_norm_file=''):
        """Initialize a mean and variance normalizer.

        Args:
            acoustic_norm_file: If given, will load from this file.
        """
        if acoustic_norm_file is '':
            self.mean_vector = None
            self.std_vector = None
        else:
            self.load_mean_std_values(acoustic_norm_file)

    def feature_normalization(self, features):
        """Normalize the input features to zero mean and unit variance.

        Args:
            features: A T*D numpy array. Acoustic feature vectors.

        Returns:
            norm_features: Normalized features.
        """
        if self.mean_vector is None:
            raise ValueError('The mean vector is not computed.')
        if self.std_vector is None:
            raise ValueError('The std vector is not computed.')
        current_frame_number = features.shape[0]
        mean_matrix = np.tile(self.mean_vector, (current_frame_number, 1))
        std_matrix = np.tile(self.std_vector, (current_frame_number, 1))
        norm_features = (features - mean_matrix) / std_matrix
        return norm_features

    def feature_denormalization(self, features):
        """Re-scale to have the original mean and variance.

        Args:
            features: A T*D numpy array. Normalized features.

        Returns:
            denorm_features: Recovered feature vectors.
        """
        if self.mean_vector is None:
            raise ValueError('The mean vector is not computed.')
        if self.std_vector is None:
            raise ValueError('The std vector is not computed.')
        current_frame_number = features.shape[0]
        mean_matrix = np.tile(self.mean_vector, (current_frame_number, 1))
        std_matrix = np.tile(self.std_vector, (current_frame_number, 1))
        denorm_features = features * std_matrix + mean_matrix
        return denorm_features

    def save_mean_std_values(self, acoutic_norm_file):
        """Save the computed mean and std vectors to a pickle file.

        Args:
            acoutic_norm_file: The path to a pickle file.
        """
        if self.mean_vector is None:
            raise ValueError('The mean vector is not computed.')
        if self.std_vector is None:
            raise ValueError('The std vector is not computed.')
        with open(acoutic_norm_file, 'wb') as writer:
            pickle.dump([self.mean_vector, self.std_vector], writer)

    def load_mean_std_values(self, acoustic_norm_file):
        """Load pre-computed mean and variance vectors.

        Args:
            acoustic_norm_file: A pickle file.

        Returns:
            The mean and std vectors.
        """
        with open(acoustic_norm_file, 'rb') as reader:
            data = pickle.load(reader)
        self.mean_vector = data[0]
        self.std_vector = data[1]
        return self.mean_vector, self.std_vector

    def compute_mean(self, acoustic_sequences):
        """Compute the mean of the acoustic features.

        Args:
            acoustic_sequences: An array of acoustic feature sequences.

        Returns:
            mean_vector: The mean vector.
        """
        num_dims = acoustic_sequences[0].shape[1]
        mean_vector = np.zeros((1, num_dims))
        all_frame_number = 0

        for seq in acoustic_sequences:
            current_frame_number = seq.shape[0]
            mean_vector += np.reshape(np.sum(seq, axis=0), (1, num_dims))
            all_frame_number += current_frame_number
        mean_vector /= float(all_frame_number)
        self.mean_vector = mean_vector
        return mean_vector

    def compute_std(self, acoustic_sequences):
        """Compute the standard deviation of the acoustic features.

        Args:
            acoustic_sequences: An array of acoustic feature sequences.

        Returns:
            std_vector: The std vector.
        """
        num_dims = acoustic_sequences[0].shape[1]
        std_vector = np.zeros((1, num_dims))
        all_frame_number = 0
        if self.mean_vector is None:
            raise ValueError('The mean vector is not computed.')

        for seq in acoustic_sequences:
            current_frame_number = seq.shape[0]
            mean_matrix = np.tile(self.mean_vector, (current_frame_number, 1))
            std_vector += np.reshape(np.sum(
                (seq - mean_matrix) ** 2, axis=0), (1, num_dims))
            all_frame_number += current_frame_number
        std_vector /= float(all_frame_number)
        std_vector = std_vector ** 0.5
        self.std_vector = std_vector
        return std_vector


class PPGSpeechLoader(torch.utils.data.Dataset):
    """Loads [ppg,speech] pairs.
    """
    def __init__(self, data_utterance_paths, hparams):
        """Data loader for the PPG->Speech task.

        Args:
            data_utterance_paths: A text file containing a list of file paths.
            hparams: The hyper-parameters.
        """
        self.data_utterance_paths = load_filepaths(data_utterance_paths)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.sequence_level = hparams.sequence_level
        self.is_skip_sil = hparams.is_skip_sil
        random.seed(hparams.seed)
        random.shuffle(self.data_utterance_paths)
        self.ppg_sequences = []
        self.acoustic_sequences = []
        for utterance_path in self.data_utterance_paths:
            feat_pairs = self.chop_utterance(utterance_path,
                                             self.sequence_level,
                                             self.is_skip_sil)
            for each_pair in feat_pairs:
                self.ppg_sequences.append(each_pair[0])
                self.acoustic_sequences.append(each_pair[1])
        self.mvn_normalizer = MeanVarianceNorm()
        if os.path.isfile(hparams.mvn_stats_file):
            self.mvn_normalizer.load_mean_std_values(hparams.mvn_stats_file)
        else:
            self.mvn_normalizer.compute_mean(self.acoustic_sequences)
            self.mvn_normalizer.compute_std(self.acoustic_sequences)

    def stack_acoustic_feats(self, utt):
        """Stack the vocoder features together to form the acoustic features.

        Stack in the form of [mcep, log_f0, band_ap].

        Args:
            utt: A data utterance object.

        Returns:
            acoustic_feat: Stacked features.
        """
        f0 = utt.f0
        lf0 = np.log(f0 + np.finfo(float).eps)  # Log F0.
        lf0 = lf0.reshape(lf0.shape[0], 1)  # Convert to 2-dim matrix.
        mcep = utt.mcep
        bap = utt.bap
        acoustic_feat = np.concatenate((mcep, lf0, bap), axis=1)
        return acoustic_feat

    def chop_following_alignment(self, ppg, acoustics, align,
                                 is_skip_sil=False):
        """Chop the given features into segments according to the alignment.

        Args:
            ppg: A T*D1 matrix, the phonetic posteriorgram.
            acoustics: A T*D2 matrix, the stacked acoustic features.
            align: A IntervalTier. Contains the alignment.
            is_skip_sil: If set to true, will skip the silences.

        Returns:
            res: A list of [ppg, acoustic_feature] pairs for each segment.
        """
        # First fix the mismatch between the ppg and acoustic frames.
        num_ppg_frames = ppg.shape[0]
        num_acoustic_frames = acoustics.shape[0]
        final_num_frames = min([num_ppg_frames, num_acoustic_frames])
        ppg = ppg[:final_num_frames, :]
        acoustics = acoustics[:final_num_frames, :]
        align.intervals[-1].maxTime = final_num_frames - 1
        res = []
        for interval in align.intervals:
            start_idx = int(interval.minTime)
            end_idx = int(interval.maxTime)
            is_last_seg = end_idx == int(final_num_frames - 1)

            if is_sil(interval.mark) and is_skip_sil:
                continue
            else:
                # The last segment should include the last frame
                if is_last_seg:
                    res.append([ppg[start_idx:(end_idx + 1), :],
                                acoustics[start_idx:(end_idx + 1), :]])
                else:
                    res.append([ppg[start_idx:end_idx, :],
                                acoustics[start_idx:end_idx, :]])
        return res

    def chop_utterance(self, data_utterance_path, level='sentence',
                       is_skip_sil=False):
        """Chop an utterances into segments.

        Args:
            data_utterance_path: The path to the data utterance protocol buffer.
            level: Level of the segmentation.
                - 'sentence': Return the whole sentence.
                - 'word': Return the word segments.
                - 'phone': Return the phone segments.
            is_skip_sil: If set to true, will skip the silences in the
            utterance. It does not take any effect if the level is 'sentence'.

        Returns:
            feat_pairs: A list, each is a [pps, acoustic_feature] pair.
        """
        utt = Utterance()
        utt.read(data_utterance_path)
        acoustic_feats = self.stack_acoustic_feats(utt)
        if level == 'sentence':
            return [[utt.monophone_ppg, acoustic_feats]]
        elif level == 'word':
            feat_pairs = self.chop_following_alignment(
                utt.monophone_ppg, acoustic_feats, utt.word, is_skip_sil)
        elif level == 'phone':
            feat_pairs = self.chop_following_alignment(
                utt.monophone_ppg, acoustic_feats, utt.phone, is_skip_sil)
        else:
            raise ValueError('Level %s is not supported.' % level)
        return feat_pairs

    def __getitem__(self, index):
        """Get a new data sample in torch.float32 format.

        Args:
            index: An int.

        Returns:
            T*D1 PPG sequence, T*D2 mvn-ed acoustic features
        """
        curr_acoustics = self.mvn_normalizer.feature_normalization(
            self.acoustic_sequences[index])
        return torch.from_numpy(self.ppg_sequences[index]).float(), \
            torch.from_numpy(curr_acoustics).float()

    def __len__(self):
        return len(self.ppg_sequences)


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
    """Loads [ppg, mel] pairs.
    """

    def __init__(self, data_utterance_paths, hparams, n_jobs=1):
        """Data loader for the PPG->Mel task.

        Args:
            data_utterance_paths: A text file containing a list of file paths.
            hparams: The hyper-parameters.
        """
        self.data_utterance_paths = load_filepaths(data_utterance_paths)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.sequence_level = hparams.sequence_level
        self.is_skip_sil = hparams.is_skip_sil
        self.is_full_ppg = hparams.is_full_ppg
        self.is_append_f0 = hparams.is_append_f0
        self.is_cache_feats = hparams.is_cache_feats
        self.load_feats_from_disk = hparams.load_feats_from_disk
        self.feats_cache_path = hparams.feats_cache_path
        self.ppg_subsampling_factor = hparams.ppg_subsampling_factor
        self.get_ppg_on_the_fly = hparams.get_ppg_on_the_fly
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
                feat_pairs = self.chop_utterance(utterance_path,
                                                 self.sequence_level,
                                                 self.is_skip_sil,
                                                 self.is_full_ppg,
                                                 self.get_ppg_on_the_fly)
                for each_pair in feat_pairs:
                    self.ppg_sequences.append(each_pair[0].astype(
                        np.float32))
                    self.acoustic_sequences.append(each_pair[1])
        if self.is_cache_feats:
            print('Caching data to %s.' % self.feats_cache_path)
            with open(self.feats_cache_path, 'wb') as f:
                pickle.dump([self.ppg_sequences, self.acoustic_sequences], f)

    def chop_following_alignment(self, ppg, acoustics, align,
                                 is_skip_sil=False):
        """Chop the given features into segments according to the alignment.

        Args:
            ppg: A T*D1 matrix, the phonetic posteriorgram.
            acoustics: A T*D2 matrix, the stacked acoustic features.
            align: A IntervalTier. Contains the alignment.
            is_skip_sil: If set to true, will skip the silences.

        Returns:
            res: A list of [ppg, acoustic_feature] pairs for each segment.
        """
        # First fix the mismatch between the ppg and acoustic frames.
        num_ppg_frames = ppg.shape[0]
        num_acoustic_frames = acoustics.shape[0]
        final_num_frames = min([num_ppg_frames, num_acoustic_frames])
        ppg = ppg[:final_num_frames, :]
        acoustics = acoustics[:final_num_frames, :]
        align.intervals[-1].maxTime = final_num_frames - 1
        res = []
        for interval in align.intervals:
            start_idx = int(interval.minTime)
            end_idx = int(interval.maxTime)
            is_last_seg = end_idx == int(final_num_frames - 1)

            if is_sil(interval.mark) and is_skip_sil:
                continue
            else:
                # The last segment should include the last frame
                if is_last_seg:
                    res.append([ppg[start_idx:(end_idx + 1), :],
                                acoustics[start_idx:(end_idx + 1), :]])
                else:
                    res.append([ppg[start_idx:end_idx, :],
                                acoustics[start_idx:end_idx, :]])
        return res

    def chop_utterance(self, data_utterance_path, level='sentence',
                       is_skip_sil=False, is_full_ppg=False,
                       get_ppg_on_the_fly=False):
        """Chop an utterances into segments.

        Args:
            data_utterance_path: The path to the data utterance protocol buffer.
            level: Level of the segmentation.
                - 'sentence': Return the whole sentence.
                - 'word': Return the word segments.
                - 'phone': Return the phone segments.
            is_skip_sil: If set to true, will skip the silences in the
            utterance. It does not take any effect if the level is 'sentence'.
            is_full_ppg: If True, will use the full PPGs.

        Returns:
            feat_pairs: A list, each is a [pps, mel] pair.
        """
        utt = Utterance()
        if get_ppg_on_the_fly:
            fs, wav = wavfile.read(data_utterance_path)
            utt.fs = fs
            utt.wav = wav
            utt.ppg = get_ppg(data_utterance_path, self.ppg_deps)
        else:
            utt.read(data_utterance_path)

        audio = torch.FloatTensor(utt.wav.astype(np.float32))
        fs = utt.fs

        if fs != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
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

        if level == 'sentence':
            if is_full_ppg:
                if self.is_append_f0:
                    ppg_f0 = append_ppg(utt.ppg, utt.f0)
                    return [[ppg_f0, acoustic_feats]]
                else:
                    return [[utt.ppg, acoustic_feats]]
            else:
                if self.is_append_f0:
                    ppg_f0 = append_ppg(utt.monophone_ppg, utt.f0)
                    return [[ppg_f0, acoustic_feats]]
                else:
                    return [[utt.monophone_ppg, acoustic_feats]]
        elif level == 'word':
            if is_full_ppg:
                feat_pairs = self.chop_following_alignment(
                    utt.ppg, acoustic_feats, utt.word, is_skip_sil)
            else:
                feat_pairs = self.chop_following_alignment(
                    utt.monophone_ppg, acoustic_feats, utt.word, is_skip_sil)
        elif level == 'phone':
            if is_full_ppg:
                feat_pairs = self.chop_following_alignment(
                    utt.ppg, acoustic_feats, utt.phone, is_skip_sil)
            else:
                feat_pairs = self.chop_following_alignment(
                    utt.monophone_ppg, acoustic_feats, utt.phone, is_skip_sil)
        else:
            raise ValueError('Level %s is not supported.' % level)
        return feat_pairs

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


class PPGMelLoaderOnDemand(torch.utils.data.Dataset):
    """Loads [ppg, mel] pairs.
    """

    def __init__(self, data_utterance_paths, hparams):
        """Data loader for the PPG->Mel task.

        Args:
            data_utterance_paths: A text file containing a list of file paths.
            hparams: The hyper-parameters.
        """
        self.data_utterance_paths = load_filepaths(data_utterance_paths)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.ppg_subsampling_factor = hparams.ppg_subsampling_factor

        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_acoustic_feat_dims, hparams.sampling_rate,
            hparams.mel_fmin, hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.data_utterance_paths)
        self.cache_mel = {}
        self.cache_ppg = {}

    def load_one_utterance(self, data_utterance_path):
        """Chop an utterances into segments.

        Args:
            data_utterance_path: The path to the data utterance protocol buffer.

        Returns:
            A [pps, mel] pair.
        """

        if data_utterance_path in self.cache_ppg:
            ppg = self.cache_ppg[data_utterance_path]
        else:
            utt = Utterance()
            utt.read(data_utterance_path)
            ppg = utt.ppg
            if len(self.cache_ppg) > 200:
                self.cache_ppg.pop(random.choice(list(self.cache_ppg.keys())))
            self.cache_ppg[data_utterance_path] = ppg

        if data_utterance_path in self.cache_mel:
            acoustic_feats = self.cache_mel[data_utterance_path]
        else:
            audio = torch.FloatTensor(utt.wav.astype(np.float32))
            fs = utt.fs

            if fs != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    fs, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm,
                                                 requires_grad=False)
            # (1, n_mel_channels, T)
            acoustic_feats = self.stft.mel_spectrogram(audio_norm)
            # (n_mel_channels, T)
            acoustic_feats = torch.squeeze(acoustic_feats, 0)
            # (T, n_mel_channels)
            acoustic_feats = acoustic_feats.transpose(0, 1)
            self.cache_mel[data_utterance_path] = acoustic_feats

        return ppg, acoustic_feats

    def __getitem__(self, index):
        """Get a new data sample in torch.float32 format.

        Args:
            index: An int.

        Returns:
            T*D1 PPG sequence, T*D2 mels
        """
        curr_ppg, curr_acoustics = \
            self.load_one_utterance(self.data_utterance_paths[index])
        if self.ppg_subsampling_factor > 1:
            curr_ppg = curr_ppg[0::self.ppg_subsampling_factor, :]
        return torch.from_numpy(curr_ppg), curr_acoustics

    def __len__(self):
        return len(self.data_utterance_paths)


class PPG2PPGLoaderOnDemand(torch.utils.data.Dataset):
    """Loads [ppg, ppg] pairs.
    """

    def __init__(self, data_utterance_paths, hparams):
        """Data loader for the PPG->Mel task.

        Args:
            data_utterance_paths: A text file containing a list of file paths.
            hparams: The hyper-parameters.
        """
        self.data_utterance_paths = load_filepaths_and_text(
            data_utterance_paths, '|')
        self.ppg_subsampling_factor = hparams.ppg_subsampling_factor
        random.seed(hparams.seed)
        random.shuffle(self.data_utterance_paths)
        self.cache_ppg = {}
        self.is_full_ppg = hparams.is_full_ppg
        self.nnet_deps = DependenciesPPG()
        if not self.is_full_ppg:
            for each_file_pair in self.data_utterance_paths:
                self.load_one_utterance(each_file_pair)

    def load_one_utterance(self, data_utterance_path):
        """Chop an utterances into segments.

        Args:
            data_utterance_path: The path to the data utterance protocol buffer.

        Returns:
            A [pps, mel] pair.
        """

        ppg = [None, None]
        if data_utterance_path in self.cache_ppg:
            ppg = self.cache_ppg[data_utterance_path]
        else:
            utt0 = Utterance()
            utt0.read(data_utterance_path[0])
            if self.is_full_ppg:
                ppg[0] = utt0.ppg
            else:
                ppg[0] = reduce_ppg_dim(Matrix(utt0.ppg),
                                        self.nnet_deps.monophone_trans).numpy()
            utt1 = Utterance()
            utt1.read(data_utterance_path[1])
            if self.is_full_ppg:
                ppg[1] = utt1.ppg
            else:
                ppg[1] = reduce_ppg_dim(Matrix(utt1.ppg),
                                        self.nnet_deps.monophone_trans).numpy()
            if self.is_full_ppg:
                if len(self.cache_ppg) > 200:
                    self.cache_ppg.pop(
                        random.choice(list(self.cache_ppg.keys())))
            self.cache_ppg[data_utterance_path] = ppg

        return ppg[0], ppg[1]

    def __getitem__(self, index):
        """Get a new data sample in torch.float32 format.

        Args:
            index: An int.

        Returns:
            T*D1 PPG sequence, T*D2 mels
        """
        curr_ppg0, curr_ppg1 = \
            self.load_one_utterance(self.data_utterance_paths[index])
        if self.ppg_subsampling_factor > 1:
            curr_ppg0 = curr_ppg0[0::self.ppg_subsampling_factor, :]
            curr_ppg1 = curr_ppg1[0::self.ppg_subsampling_factor, :]
        return torch.from_numpy(curr_ppg0), torch.from_numpy(curr_ppg1)

    def __len__(self):
        return len(self.data_utterance_paths)


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
