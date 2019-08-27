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
from common import feat
from scipy.io import wavfile
from kaldi.util.io import read_matrix
from kaldi.feat.functions import splice_frames


class TestFeatFunctions(unittest.TestCase):
    def setUp(self):
        self.mono_wav = "data/test_mono_channel.wav"
        self.dual_wav = "data/test_dual_channel.wav"
        self.wave_data = feat.read_wav_kaldi(self.mono_wav)

    def tearDown(self):
        pass

    def test_read_wav_internal(self):
        fs, wav = wavfile.read(self.mono_wav, False)
        wave_data = feat.read_wav_kaldi_internal(wav, fs)
        # Correct number of channels
        self.assertEqual(wave_data.data().num_rows, 1)
        # Correct number of samples
        self.assertEqual(wav.shape[0], wave_data.data().num_cols)

    def test_read_mono_wav(self):
        fs, wav = wavfile.read(self.mono_wav, False)
        wave_data = feat.read_wav_kaldi(self.mono_wav)
        # Correct number of channels
        self.assertEqual(wave_data.data().num_rows, 1)
        # Correct number of samples
        self.assertEqual(wav.shape[0], wave_data.data().num_cols)

    def test_read_dual_wav(self):
        fs, wav = wavfile.read(self.dual_wav, False)
        wave_data = feat.read_wav_kaldi(self.dual_wav)
        # Correct number of channels, we only keep the first one
        self.assertEqual(wave_data.data().num_rows, 1)
        # Correct number of samples
        self.assertEqual(wav.shape[0], wave_data.data().num_cols)

    def test_compute_mfcc(self):
        mfcc_opts = feat.MfccOptions()
        mfcc_opts.frame_opts.allow_downsample = True
        mfcc_opts.frame_opts.snip_edges = False
        mfccs = feat.compute_mfcc_feats(self.wave_data, mfcc_opts)
        self.assertEqual(mfccs.num_cols, 13)  # Default MFCC dims
        expected_num_frames = self.wave_data.data().num_cols / (
                self.wave_data.samp_freq * mfcc_opts.frame_opts.frame_shift_ms / 1000)
        expected_num_frames = int(round(expected_num_frames))  # Closest int
        self.assertEqual(mfccs.num_rows, expected_num_frames)

    def test_cmn(self):
        mfcc_opts = feat.MfccOptions()
        mfcc_opts.frame_opts.allow_downsample = True
        mfcc_opts.frame_opts.snip_edges = False
        mfccs = feat.compute_mfcc_feats(self.wave_data, mfcc_opts)
        mfcc_cmn = feat.apply_cepstral_mean_norm(mfccs)
        self.assertAlmostEqual(mfcc_cmn.sum(), 0, 2)

    def test_lda_trans(self):
        mfcc_opts = feat.MfccOptions()
        mfcc_opts.frame_opts.allow_downsample = True
        mfcc_opts.frame_opts.snip_edges = False
        mfccs = feat.compute_mfcc_feats(self.wave_data, mfcc_opts)
        trans = read_matrix("data/lda.mat")
        mfccs = splice_frames(mfccs, 3, 3)
        mfcc_lda = feat.apply_feat_transform(mfccs, trans)
        self.assertEqual(mfcc_lda.num_rows, mfccs.num_rows)
        self.assertEqual(mfcc_lda.num_cols, trans.num_rows)

    def test_read_sparse_matrix(self):
        sparse_mat = feat.read_sparse_mat("data/reduce_dim.mat")
        self.assertEqual(sparse_mat.sum(), 5816)  # This is a special matrix


if __name__ == '__main__':
    unittest.main()
