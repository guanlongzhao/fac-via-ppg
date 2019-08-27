# Copyright 2018 Guanlong Zhao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module provides handy functions to perform Kaldi-related feature
operations.
"""

import logging
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.feat.wave import WaveData
from kaldi.matrix import Vector, Matrix, SubMatrix
from kaldi.matrix.common import MatrixTransposeType
from kaldi.matrix.sparse import SparseMatrix
from kaldi.util.io import xopen
from kaldi import online2
from scipy.io import wavfile


def read_wav_kaldi_internal(wav, fs) -> WaveData:
    """Internal function for converting wave data to Kaldi format.

    This function will only keep the first channel.

    Args:
        wav: S*C ndarray. S is number of samples and C is number of channels.
        fs: Sampling frequency.

    Returns:
        wd: A Kaldi-readable WaveData object.
    """
    # Only keep the first channel if more than one
    if wav.ndim >= 2:
        wav = wav[:, 0]

    # Save to a Kaldi matrix, per Kaldi's requirement.
    wav_kaldi = Matrix(1, len(wav))
    wav_kaldi.copy_rows_from_vec_(Vector(wav))

    if hasattr(WaveData, 'new'):
        wd = WaveData.new(fs, wav_kaldi)
    elif hasattr(WaveData, 'from_data'):
        wd = WaveData.from_data(fs, wav_kaldi)
    else:
        wd = None
        logging.error('Unknown Pykaldi package.')
    return wd


def read_wav_kaldi(wav_file_path: str) -> WaveData:
    """Read a given wave file to a Kaldi readable format.

    Args:
        wav_file_path: Path to a .wav file.
    Returns:
        wd: A Kaldi-readable WaveData object.
    """
    # Read in as np array not memmap.
    fs, wav = wavfile.read(wav_file_path, False)

    wd = read_wav_kaldi_internal(wav, fs)
    return wd


def compute_mfcc_feats(wav: WaveData, mfcc_opts: MfccOptions) -> Matrix:
    """Compute MFCC features given a Kaldi WaveData.

    Args:
        wav: A WaveData object.
        mfcc_opts: An MfccOptions object containing feature extraction options.
        A few notable options are,
        - use_energy: Generally I will use False, since the energy does not
        contain much linguistic information
        - frame_opts.allow_downsample: Generally I will set this to True, since
        the AM I use can only handle the default sampling frequency (16KHz)
        - frame_opts.frame_shift_ms: For speech synthesis purposes, might be
        good to have a smaller shift (e.g., 5ms)
        - frame_opts.snip_edges: Generally I will set this to False, just to
        have a deterministic way to compute the number of frames

    Returns:
        feats: A T*D MFCC feature matrix.
    """
    mfcc = Mfcc(mfcc_opts)
    vtln_warp = 1.0  # This is the default value
    channel = 0  # Only use the first channel

    feats = mfcc.compute_features(wav.data()[channel], wav.samp_freq, vtln_warp)
    return feats


def apply_cepstral_mean_norm(feats: Matrix) -> Matrix:
    """Apply cepstral mean normalization to MFCCs.

    Note that this function does not do variance normalization, which is enough
    for GZ's purposes.

    Args:
        feats: A T*D MFCC features.

    Returns:
        feats: A T*D Normalized MFCCs.
    """
    mean = Vector(feats.num_cols)
    mean.add_row_sum_mat_(1.0, feats)
    mean.scale_(1.0 / feats.num_rows)
    for i in range(feats.num_rows):
        feats[i].add_vec_(-1.0, mean)
    return feats


def apply_feat_transform(feats: Matrix, transform: Matrix) -> Matrix:
    """Apply an LDA/fMLLR transform on the input features.

    The transform is a simple matrix multiplication: F = FT' (' is transpose) in
    the case of LDA. For fMLLR, please see
    http://kaldi-asr.org/doc/transform.html#transform_cmllr_global
    This function is an extremely simplified version of
    https://github.com/kaldi-asr/kaldi/blob/5.3/src/featbin/transform-feats.cc

    Args:
        feats: A T*D feature matrix.
        transform: A D'*D matrix, where D' is the output feature dim.

    Returns:
        feats_out: A T*D' matrix.
    """
    feat_dim = feats.num_cols
    transform_rows = transform.num_rows
    transform_cols = transform.num_cols

    feats_out = Matrix(feats.num_rows, transform_rows)
    if transform_cols == feat_dim:
        feats_out.add_mat_mat_(feats, transform, MatrixTransposeType.NO_TRANS,
                               MatrixTransposeType.TRANS, 1.0, 0.0)
    elif transform_cols == feat_dim + 1:
        # Append the implicit 1.0 to the input feature.
        linear_part = SubMatrix(transform, 0, transform_rows, 0, feat_dim)
        feats_out.add_mat_mat_(feats, linear_part, MatrixTransposeType.NO_TRANS,
                               MatrixTransposeType.TRANS, 1.0, 0.0)
        offset = Vector(transform_rows)
        offset.copy_col_from_mat_(transform, feat_dim)
        feats_out.add_vec_to_rows_(1.0, offset)
    else:
        logging.error(("Transform matrix has bad dimension %dx%d versus feat "
                       "dim %d") % (transform_rows, transform_cols, feat_dim))
    return feats_out


def read_sparse_mat(sparse_mat_dir: str) -> SparseMatrix:
    """Read in a sparse matrix.

    Args:
        sparse_mat_dir: Path to the sparse matrix file.

    Returns:
        mat: A sparse matrix.
    """
    with xopen(sparse_mat_dir, 'r') as reader:
        mat = SparseMatrix()
        mat.read_(reader.stream(), reader.binary)
    return mat


def parse_config(config_path: str):
    """Parse a very simple config file. Each line is an option in the form of
    "--name=value".

    Args:
        config_path: Path to the config file.

    Returns:
        A dict whose keys are "--name" and values are "value".
    """
    with open(config_path, 'r') as f:
        # This is a legit syntax.
        options = dict([tuple(line.split('=')) for
                        line in f.read().splitlines(False)])
    return options


def compute_mfcc_plus_ivector(wav_data: WaveData, config_path: str):
    """Compute MFCCs and ivector using the OnlineNnetFeaturePipeline.
    Inspired by https://github.com/kaldi-asr/kaldi/blob/master/src/online2bin
    /online2-wav-nnet3-latgen-faster.cc

    Args:
        wav_data: WaveData.
        config_path: Path to the online config file.

    Returns:
        mfccs: T*D1 matrix.
        ivectors: T*D2 matrix.
    """
    options = parse_config(config_path)
    feature_opts = online2.OnlineNnetFeaturePipelineConfig()
    feature_opts.ivector_extraction_config = \
        options['--ivector-extraction-config']
    feature_opts.mfcc_config = options['--mfcc-config']
    feature_opts.silence_weighting_config.silence_phones_str = \
        options['--endpoint.silence-phones']
    feature_info = \
        online2.OnlineNnetFeaturePipelineInfo().from_config(feature_opts)
    feature_info.ivector_extractor_info.use_most_recent_ivector = True
    feature_info.ivector_extractor_info.greedy_ivector_extractor = True
    feature_info.mfcc_opts.frame_opts.allow_downsample = True
    feature_info.mfcc_opts.frame_opts.frame_shift_ms = \
        int(options['--frame-shift'])
    feature_info.ivector_extractor_info.ivector_period = \
        int(options['--frame-shift'])
    feature_pipeline = online2.OnlineNnetFeaturePipeline(feature_info)

    wav = \
        Vector(wav_data.data().shape[1]).copy_row_from_mat_(wav_data.data(), 0)

    feature_pipeline.accept_waveform(wav_data.samp_freq, wav)
    feature_pipeline.input_finished()
    num_frames = feature_pipeline.num_frames_ready()
    num_ceps = feature_info.mfcc_opts.num_ceps
    mfccs = Matrix(num_frames, num_ceps)
    feature_pipeline.input_feature().get_frames(list(range(num_frames)), mfccs)

    num_ivecs = feature_info.ivector_extractor_info.extractor.ivector_dim()
    ivectors = Matrix(num_frames, num_ivecs)
    feature_pipeline.ivector_feature().get_frames(list(range(num_frames)),
                                                  ivectors)

    return mfccs, ivectors
