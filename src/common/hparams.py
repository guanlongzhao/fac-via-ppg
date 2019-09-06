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


class HParamsView(object):
    def __init__(self, d):
        self.__dict__ = d


def create_hparams(**kwargs):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = {
        ################################
        # Experiment Parameters        #
        ################################
        "epochs": 1000,
        "iters_per_checkpoint": 200,
        "seed": 16807,
        "dynamic_loss_scaling": True,
        "fp16_run": False,
        "distributed_run": False,
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "output_directory": None,  # Directory to save checkpoints.
        # Directory to save tensorboard logs. Just keep it like this.
        "log_directory": 'log',
        "checkpoint_path": '',  # Path to a checkpoint file.
        "warm_start": False,  # Load the model only (warm start)
        "n_gpus": 1,  # Number of GPUs
        "rank": 0,  # Rank of current gpu
        "group_name": 'group_name',  # Distributed group name

        ################################
        # Data Parameters             #
        ################################
        # Passed as a txt file, see data/filelists/training-set.txt for an
        # example.
        "training_files": '',
        # Passed as a txt file, see data/filelists/validation-set.txt for an
        # example.
        "validation_files": '',
        "is_full_ppg": True,  # Whether to use the full PPG or not.
        "is_append_f0": False,  # Currently only effective at sentence level
        "ppg_subsampling_factor": 1,  # Sub-sample the ppg & acoustic sequence.
        # Cases
        # |'load_feats_from_disk'|'is_cache_feats'|Note
        # |True                  |True            |Error
        # |True                  |False           |Please set cache path
        # |False                 |True            |Overwrite the cache path
        # |False                 |False           |Ignores the cache path
        "load_feats_from_disk": False,  # Remember to set the path.
        # Mutually exclusive with 'load_feats_from_disk', will overwrite
        # 'feats_cache_path' if set.
        "is_cache_feats": False,
        "feats_cache_path": '',

        ################################
        # Audio Parameters             #
        ################################
        "max_wav_value": 32768.0,
        "sampling_rate": 16000,
        "n_acoustic_feat_dims": 80,
        "filter_length": 1024,
        "hop_length": 160,
        "win_length": 1024,
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,

        ################################
        # Model Parameters             #
        ################################
        "n_symbols": 5816,
        "symbols_embedding_dim": 600,

        # Encoder parameters
        "encoder_kernel_size": 5,
        "encoder_n_convolutions": 3,
        "encoder_embedding_dim": 600,

        # Decoder parameters
        "decoder_rnn_dim": 300,
        "prenet_dim": 300,
        "max_decoder_steps": 1000,
        "gate_threshold": 0.5,
        "p_attention_dropout": 0.1,
        "p_decoder_dropout": 0.1,

        # Attention parameters
        "attention_rnn_dim": 300,
        "attention_dim": 150,
        # +- time steps to look at when computing the attention. Set to None
        # to block it.
        "attention_window_size": 20,

        # Location Layer parameters
        "attention_location_n_filters": 32,
        "attention_location_kernel_size": 31,

        # Mel-post processing network parameters
        "postnet_embedding_dim": 512,
        "postnet_kernel_size": 5,
        "postnet_n_convolutions": 5,

        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate": False,
        "learning_rate": 1e-5,
        "weight_decay": 1e-6,
        "grad_clip_thresh": 1.0,
        "batch_size": 6,
        "mask_padding": True,  # set model's padded outputs to padded values
        "mel_weight": 1,
        "gate_weight": 0.005
    }

    for key, val in kwargs.items():
        if key in hparams:
            hparams[key] = val
        else:
            raise ValueError('The hyper-parameter %s is not supported.' % key)

    hparams_view = HParamsView(hparams)

    return hparams_view


def create_hparams_stage(**kwargs):
    """Create model hyperparameters. Parse nondefault from given string.

    These are the parameters used for our interspeech 2019 submission.
    """

    hparams = {
        'attention_dim': 150,
        'attention_location_kernel_size': 31,
        'attention_location_n_filters': 32,
        'attention_rnn_dim': 300,
        'attention_window_size': 20,
        'batch_size': 6,
        'checkpoint_path': None,
        'cudnn_benchmark': False,
        'cudnn_enabled': True,
        'decoder_rnn_dim': 300,
        'dist_backend': 'nccl',
        'dist_url': 'tcp://localhost:54321',
        'distributed_run': False,
        'dynamic_loss_scaling': True,
        'encoder_embedding_dim': 600,
        'encoder_kernel_size': 5,
        'encoder_n_convolutions': 3,
        'epochs': 1000,
        'feats_cache_path': '',
        'filter_length': 1024,
        'fp16_run': False,
        'gate_threshold': 0.5,
        'gate_weight': 0.005,
        'grad_clip_thresh': 1.0,
        'group_name': 'group_name',
        'hop_length': 160,
        'is_append_f0': False,
        'is_cache_feats': False,
        'is_full_ppg': True,
        'is_large_set': False,
        'is_skip_sil': False,
        'iters_per_checkpoint': 100,
        'learning_rate': 0.0001,
        'load_feats_from_disk': True,
        'log_directory': 'log',
        'mask_padding': True,
        'max_decoder_steps': 1000,
        'max_wav_value': 32768.0,
        'mel_fmax': 8000.0,
        'mel_fmin': 0.0,
        'mel_weight': 1,
        'mvn_stats_file': '',
        'n_acoustic_feat_dims': 80,
        'n_gpus': 1,
        'n_symbols': 5816,
        'output_directory': '',
        'p_attention_dropout': 0.1,
        'p_decoder_dropout': 0.1,
        'postnet_embedding_dim': 512,
        'postnet_kernel_size': 5,
        'postnet_n_convolutions': 5,
        'ppg_subsampling_factor': 1,
        'prenet_dim': 300,
        'rank': 0,
        'sampling_rate': 16000,
        'seed': 16807,
        'sequence_level': 'sentence',
        'symbols_embedding_dim': 600,
        'training_files': '',
        'use_saved_learning_rate': False,
        'validation_files': '',
        'warm_start': False,
        'weight_decay': 1e-06,
        'win_length': 1024}

    for key, val in kwargs.items():
        if key in hparams:
            hparams[key] = val
        else:
            raise ValueError('The hyper-parameter %s is not supported.' % key)

    hparams_view = HParamsView(hparams)

    return hparams_view
