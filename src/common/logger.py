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

"""From https://github.com/NVIDIA/tacotron2"""

import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from common.plotting_utils import plot_alignment_to_numpy, \
    plot_spectrogram_to_numpy, plot_ppg_to_numpy
from common.plotting_utils import plot_gate_outputs_to_numpy
from ppg import DependenciesPPG, reduce_ppg_dim
from kaldi.matrix import Matrix


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        mel_outputs_before_postnet, mel_outputs, gate_outputs, alignments = \
            y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration)
        self.add_image(
            "mel_predicted_before_postnet",
            plot_spectrogram_to_numpy(
                mel_outputs_before_postnet[idx].data.cpu().numpy()), iteration)
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                F.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration)


class WaveglowLogger(SummaryWriter):
    def __init__(self, logdir):
        super(WaveglowLogger, self).__init__(logdir)

    def log_training(self, reduced_loss, iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)


class PPG2PPGLogger(SummaryWriter):
    def __init__(self, logdir, is_full_ppg=True):
        super(PPG2PPGLogger, self).__init__(logdir, is_full_ppg)
        self.nnet_deps = DependenciesPPG()
        self.is_full_ppg = is_full_ppg

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration,
                       lengths):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        t = lengths[idx]
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                F.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration)
        # (B, D, T) -> (D, T) -> (T, D)
        if self.is_full_ppg:
            ppg_tgt_full = Matrix(mel_targets[idx, :, 0:t].data.cpu().numpy().T)
            ppg_tgt = reduce_ppg_dim(ppg_tgt_full,
                                     self.nnet_deps.monophone_trans)
            ppg_tgt = ppg_tgt.numpy()
        else:
            ppg_tgt = mel_targets[idx, :, 0:t].data.cpu().numpy().T
        self.add_image(
            "mel_target",
            plot_ppg_to_numpy(ppg_tgt), iteration)
        # (B, D, T) -> (D, T) -> (T, D)
        if self.is_full_ppg:
            ppg_pred_full = Matrix(
                mel_outputs[idx, :, 0:t].exp().data.cpu().numpy().T)
            ppg_pred = reduce_ppg_dim(ppg_pred_full,
                                      self.nnet_deps.monophone_trans)
            ppg_pred = ppg_pred.numpy()
        else:
            ppg_pred = mel_outputs[idx, :, 0:t].exp().data.cpu().numpy().T
        self.add_image(
            "mel_predicted",
            plot_ppg_to_numpy(ppg_pred), iteration)
