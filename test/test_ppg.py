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

import unittest
import ppg
from common import feat, decode
from kaldi.util.io import read_matrix


class TestPPGFunctions(unittest.TestCase):
    def setUp(self):
        self.wav_path = "data/test_mono_channel.wav"
        self.nnet_path = "data/nnet3.raw"
        self.lda_path = "data/lda.mat"
        self.lda_dim = 40
        self.reduce_dim_trans_path = "data/reduce_dim.mat"

    def tearDown(self):
        pass

    def test_compute_feat_for_nnet(self):
        feats = ppg.compute_feat_for_nnet(self.wav_path, self.lda_path)
        self.assertEqual(feats.num_cols, self.lda_dim)

    def test_compute_feat_for_nnet_internal(self):
        wave_data = feat.read_wav_kaldi(self.wav_path)
        trans = read_matrix(self.lda_path)
        shift = 10
        feats = ppg.compute_feat_for_nnet_internal(wave_data, trans,
                                                   frame_shift=shift)
        expected_num_frames = wave_data.data().num_cols / (
                wave_data.samp_freq * shift / 1000)
        expected_num_frames = int(round(expected_num_frames))  # Closest int
        self.assertEqual(feats.num_rows, expected_num_frames)
        self.assertEqual(feats.num_cols, self.lda_dim)

    def test_compute_full_ppg(self):
        nnet = decode.read_nnet3_model(self.nnet_path)
        feats = ppg.compute_feat_for_nnet(self.wav_path, self.lda_path)
        raw_ppgs = ppg.compute_full_ppg(nnet, feats)
        ppg_dims = 5816
        self.assertEqual(raw_ppgs.num_cols, ppg_dims)
        self.assertAlmostEqual(raw_ppgs.sum(), feats.num_rows, 1)

    def test_reduce_ppg_dim(self):
        nnet = decode.read_nnet3_model(self.nnet_path)
        feats = ppg.compute_feat_for_nnet(self.wav_path, self.lda_path)
        raw_ppgs = ppg.compute_full_ppg(nnet, feats)
        reduce_dim_trans = feat.read_sparse_mat(self.reduce_dim_trans_path)
        reduce_ppg = ppg.reduce_ppg_dim(raw_ppgs, reduce_dim_trans)
        reduce_ppg_dim = reduce_dim_trans.num_rows
        self.assertEqual(reduce_ppg.num_cols, reduce_ppg_dim)
        self.assertAlmostEqual(reduce_ppg.sum(), feats.num_rows, 1)

    def test_compute_monophone_ppg(self):
        deps = ppg.DependenciesPPG()
        wave_data = feat.read_wav_kaldi(self.wav_path)
        ppgs = ppg.compute_monophone_ppg(wave_data, deps.nnet, deps.lda,
                                         deps.monophone_trans)
        reduce_ppg_dim = deps.monophone_trans.num_rows
        self.assertEqual(ppgs.shape[1], reduce_ppg_dim)
        self.assertAlmostEqual(ppgs.sum(), ppgs.shape[0], 1)

    def test_ppg_dependencies(self):
        deps = ppg.DependenciesPPG()
        self.assertIsNotNone(deps)
