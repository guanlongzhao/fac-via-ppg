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

"""This module contains functions related to PPGs.
"""

import os
import re
import logging
from kaldi import nnet3, chain
from kaldi.nnet3 import Nnet
from kaldi.util.io import read_matrix
from kaldi.feat.functions import splice_frames
from kaldi.feat.wave import WaveData
from kaldi.feat.mfcc import Mfcc, MfccOptions
from kaldi.matrix.sparse import SparseMatrix
from kaldi.matrix import Matrix, Vector
from kaldi.matrix.common import MatrixTransposeType
from kaldi.cudamatrix import CuMatrix
from kaldi.fstext import StdVectorFst
from math import isnan
from common import feat
from common import decode
from numpy import ndarray

# Static resources
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..',
                        'data')
NNET_PATH = os.path.join(DATA_DIR, 'am', 'final.raw')
LDA_PATH = os.path.join(DATA_DIR, 'feats', 'final.mat')
REDUCE_DIM_PATH = os.path.join(DATA_DIR, 'feats', 'reduce_dim.mat')
SPLICE_OPTS_PATH = os.path.join(DATA_DIR, 'feats', 'splice_opts')
# Speaker dependent fmllr transforms. key: speaker-id, value: path to fmllr
# matrix file.
FMLLR_PATHS = {'rms': os.path.join(DATA_DIR, 'feats', 'rms.fmllr'),
               'clb': os.path.join(DATA_DIR, 'feats', 'clb.fmllr')}


def compute_full_ppg(nnet: Nnet, feats: Matrix) -> Matrix:
    """Compute full PPG features given appropriate input features.

    Args:
        nnet: An neural network AM.
        feats: Suitable T*D input feature matrix.

    Returns:
        raw_ppgs: T*K raw PPGs, K is the number of senones.
    """
    # Obtain the nnet computer, for some unknown reason, the computer must be
    # constructed within this function.
    nnet3.set_batchnorm_test_mode(True, nnet)
    nnet3.set_dropout_test_mode(True, nnet)
    nnet3.collapse_model(nnet3.CollapseModelConfig(), nnet)
    opts = nnet3.NnetSimpleComputationOptions()
    opts.acoustic_scale = 1.0
    compiler = nnet3.CachingOptimizingCompiler. \
        new_with_optimize_opts(nnet, opts.optimize_config)
    priors = Vector()  # We do not need prior
    nnet_computer = nnet3.DecodableNnetSimple(opts, nnet, priors, feats,
                                              compiler)
    # Obtain frame-level PPGs
    raw_ppgs = Matrix(nnet_computer.num_frames(), nnet_computer.output_dim())
    for i in range(nnet_computer.num_frames()):
        temp = Vector(nnet_computer.output_dim())
        nnet_computer.get_output_for_frame(i, temp)
        raw_ppgs.copy_row_from_vec_(temp, i)
    return raw_ppgs


def compute_chain_full_ppg(nnet: Nnet, den_fst: StdVectorFst, mfccs: Matrix,
                           ivecs: Matrix, is_hidden_feats=False) -> Matrix:
    """Compute full PPG features given appropriate input features using a
    chain model. It can be a BiLSTM or TDNN model.

    It is a close reimplementation of
    https://github.com/kaldi-asr/kaldi/blob/master/src/chainbin/nnet3-chain
    -compute-post.cc

    Args:
        nnet: An neural network AM.
        den_fst: A dense FST.
        mfccs: Suitable T*D input feature matrix.
        ivecs: T*D' ivector matrix.
        is_hidden_feats: If set to True, it means that the given nnet model
        will output the hidden bottleneck features -- the nnet output prior
        to the softmax (or other operations). And we will skip the chain
        operations.

    Returns:
        posteriors_cpu: T*K raw PPGs, K is the number of senones.
    """
    # Obtain the nnet computer, for some unknown reason, the computer must be
    # constructed within this function.
    nnet3.set_batchnorm_test_mode(True, nnet)
    nnet3.set_dropout_test_mode(True, nnet)
    nnet3.collapse_model(nnet3.CollapseModelConfig(), nnet)
    opts = nnet3.NnetSimpleComputationOptions()
    opts.acoustic_scale = 1.0
    compiler = nnet3.CachingOptimizingCompiler. \
        new_with_optimize_opts(nnet, opts.optimize_config)
    priors = Vector()  # We do not need prior
    nnet_computer = nnet3.DecodableNnetSimple(opts, nnet, priors, mfccs,
                                              compiler, None, ivecs, 1)
    # Obtain frame-level raw nnet outputs -- they are not PPGs yet.
    raw_ppgs = Matrix(nnet_computer.num_frames(), nnet_computer.output_dim())
    for i in range(nnet_computer.num_frames()):
        temp = Vector(nnet_computer.output_dim())
        nnet_computer.get_output_for_frame(i, temp)
        raw_ppgs.copy_row_from_vec_(temp, i)

    if is_hidden_feats:
        return raw_ppgs

    # Chain operations.
    chain_opts = chain.ChainTrainingOptions()
    chain_opts.leaky_hmm_coefficient = 0.1
    num_pdfs = nnet.output_dim('output')
    den_graph = chain.DenominatorGraph(den_fst, num_pdfs)
    gpu_raw_ppgs = CuMatrix.from_matrix(raw_ppgs)
    den_computation = chain.DenominatorComputation(chain_opts, den_graph, 1,
                                                   gpu_raw_ppgs)
    forward_prob = den_computation.forward()
    num_frames = gpu_raw_ppgs.num_rows()
    posteriors = CuMatrix.from_size(num_frames, num_pdfs)
    ok = den_computation.backward(1, posteriors)

    if not ok or isnan(forward_prob):
        raise ValueError('Something went wrong for this utterance;'
                         'forward-prob = %f, num-frames = %d.', forward_prob,
                         num_frames)

    posteriors_cpu = Matrix(num_frames, num_pdfs)
    posteriors.swap_with_matrix(posteriors_cpu)
    return posteriors_cpu.numpy()


def reduce_ppg_dim(ppgs: Matrix, transform: SparseMatrix) -> Matrix:
    """Reduce full PPGs to monophone PPGs.

    Args:
        ppgs: A T*D PPG matrix.
        transform: A d*D sparse matrix.

    Returns:
        monophone_ppgs: A T*d matrix. Containing PPGs reduced into monophones.
    """
    num_frames = ppgs.num_rows
    num_phones = transform.num_rows

    # Convert the sparse matrix to a full matrix to avoid having to keep the
    # matrix type consistent
    full_transform = Matrix(num_phones, transform.num_cols)
    transform.copy_to_mat(full_transform)

    monophone_ppg = Matrix(num_frames, num_phones)
    monophone_ppg.add_mat_mat_(ppgs, full_transform,
                               MatrixTransposeType.NO_TRANS,
                               MatrixTransposeType.TRANS, 1.0, 0.0)
    return monophone_ppg


def compute_feat_for_nnet_internal(wav: WaveData, lda: Matrix,
                                   **kwargs) -> Matrix:
    """This is an internal wrapper for computing input features to an AM.

    Args:
        wav: A Kaldi WaveData object.
        lda: A D*D LDA transform matrix.
        **kwargs: See "options." "fmllr" should be a D*(D+1) matrix.

    Returns:
        feats: A T*D feature matrix.
    """
    options = {"is_use_energy": False, "is_downsample": True, "frame_shift": 10,
               "is_snip_edges": False, "left_context": 3, "right_context": 3,
               "fmllr": None}
    for key, val in kwargs.items():
        if key in options:
            options[key] = val
        else:
            logging.error("Option %s not allowed!" % (key))

    # Get MFCCs
    mfcc_opts = MfccOptions()
    mfcc_opts.use_energy = options["is_use_energy"]
    mfcc_opts.frame_opts.allow_downsample = options["is_downsample"]
    mfcc_opts.frame_opts.frame_shift_ms = options["frame_shift"]
    mfcc_opts.frame_opts.snip_edges = options["is_snip_edges"]
    mfccs = feat.compute_mfcc_feats(wav, mfcc_opts)

    # Apply CMN
    mfccs = feat.apply_cepstral_mean_norm(mfccs)

    # Splice feats
    feats = splice_frames(mfccs,
                          options["left_context"], options["right_context"])

    # Apply LDA
    feats = feat.apply_feat_transform(feats, lda)

    # Apply fMLLR
    if options['fmllr']:
        feats = feat.apply_feat_transform(feats, options['fmllr'])

    return feats


def compute_feat_for_nnet(wav_path: str, lda_path: str) -> Matrix:
    """This is the external wrapper for computing input features to an AM.

    This function will not apply the fMLLR transform.

    Args:
        wav_path: Path to a wave file.
        lda_path: Path to an LDA transform matrix.

    Returns:
        feats: A T*D feature matrix.
    """
    if os.path.exists(wav_path):
        wave_data = feat.read_wav_kaldi(wav_path)
    else:
        logging.error("File %s does not exist." % (wav_path))

    if os.path.exists(lda_path):
        trans = read_matrix(lda_path)
    else:
        logging.error("Transform file %s does not exist." % (lda_path))

    feats = compute_feat_for_nnet_internal(wave_data, trans)
    return feats


def compute_monophone_ppg(wav: WaveData, nnet: Nnet, lda: Matrix,
                          transform: SparseMatrix, shift=10,
                          fmllr=None) -> ndarray:
    """A convenient one-stop interface for computing monophone PPGs.

    Args:
        wav: Speech data.
        nnet: Acoustic model.
        lda: LDA transform.
        transform: Pdf-to-Monophone transformation.
        shift: [optional] Frame shift in ms.
        fmllr: [optional] fMLLR transform matrix.

    Returns:
        monophone_ppgs: Monophone PPGs in numpy array.
    """
    feats = compute_feat_for_nnet_internal(wav, lda, frame_shift=shift,
                                           fmllr=fmllr)
    raw_ppgs = compute_full_ppg(nnet, feats)
    monophone_ppgs = reduce_ppg_dim(raw_ppgs, transform)
    monophone_ppgs = monophone_ppgs.numpy()
    return monophone_ppgs


def compute_full_ppg_wrapper(wav: WaveData, nnet: Nnet, lda: Matrix,
                             shift=10, fmllr=None) -> ndarray:
    """A convenient one-stop interface for computing the full PPGs.

    Args:
        wav: Speech data.
        nnet: Acoustic model.
        lda: LDA transform.
        shift: [optional] Frame shift in ms.
        fmllr: [optional] fMLLR transform matrix.

    Returns:
        raw_ppgs: Full PPGs in numpy array.
    """
    feats = compute_feat_for_nnet_internal(wav, lda, frame_shift=shift,
                                           fmllr=fmllr)
    raw_ppgs = compute_full_ppg(nnet, feats)
    raw_ppgs = raw_ppgs.numpy()
    return raw_ppgs


class DependenciesPPG(object):
    """Load all necessary resources for computing PPGs.

    Sample usage,
        deps = DependenciesPPG()
        nnet = deps.nnet
    """

    def __init__(self,
                 nnet_path=NNET_PATH,
                 lda_path=LDA_PATH,
                 reduce_dim_path=REDUCE_DIM_PATH,
                 splice_opts_path=SPLICE_OPTS_PATH,
                 fmllr_paths=FMLLR_PATHS):
        """Load the given resources.

        Args:
            nnet_path: Path to acoustic model.
            lda_path: Path to LDA.
            reduce_dim_path: Path to pdf-to-Monophone transformation.
            splice_opts_path: Path to splice options.
            fmllr_paths: See FMLLR_PATHS.
        """
        # Check inputs
        if not os.path.isfile(nnet_path):
            logging.error("File %s does not exist!", nnet_path)
        self.nnet_path = nnet_path
        if not os.path.isfile(lda_path):
            logging.error("File %s does not exist!", lda_path)
        self.lda_path = lda_path
        if not os.path.isfile(reduce_dim_path):
            logging.error("File %s does not exist!", reduce_dim_path)
        self.reduce_dim_path = reduce_dim_path
        if not os.path.isfile(splice_opts_path):
            logging.error("File %s does not exist!", splice_opts_path)
        self.splice_opts_path = splice_opts_path

        # Read in those dependencies
        self.context_parser = re.compile(r"--left-context=(\d+) "
                                         r"--right-context=(\d+)")
        self.nnet = decode.read_nnet3_model(nnet_path)
        self.lda = read_matrix(lda_path)
        self.monophone_trans = feat.read_sparse_mat(reduce_dim_path)
        with open(splice_opts_path, 'r') as reader:
            splice_opts = reader.readline()
        self.splice_opts = splice_opts
        if splice_opts:
            context = self.context_parser.match(splice_opts)
            context = context.groups()
        else:
            context = (None, None)
            logging.warning("Splice options are empty.")
        self.left_context = context[0]
        self.right_context = context[1]
        self.fmllr_mats = {}
        for key, val in fmllr_paths.items():
            if not os.path.isfile(val):
                logging.error("File %s does not exist!", val)
            self.fmllr_mats[key] = read_matrix(val)
