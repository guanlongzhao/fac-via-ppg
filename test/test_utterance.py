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
import tempfile
import numpy as np
import os
from common import utterance
from common.data_utterance_pb2 import FloatMatrix, Int32Matrix, BinaryMatrix, \
    Segment
from textgrid import IntervalTier, TextGrid
from scipy.io import wavfile


class TestUtterance(unittest.TestCase):
    def setUp(self):
        self.utt = utterance.Utterance()
        self.float_mat = FloatMatrix()
        self.int_mat = Int32Matrix()
        self.bool_mat = BinaryMatrix()
        self.seg = Segment()
        self.tier = IntervalTier('test', 0, 2)
        self.tier.add(0, 1, "a")
        self.tier.add(1, 2, "b")

    def tearDown(self):
        pass

    def test_empty_init(self):
        self.assertTrue(isinstance(self.utt, utterance.Utterance))

    def test_full_init(self):
        utt = utterance.Utterance(np.array([1, 2, 3, 4, 5]), 16000, "test")
        self.assertTrue(isinstance(utt, utterance.Utterance))

    def test_invalid_init(self):
        try:
            utt = utterance.Utterance(np.array([1, 2, 3, 4, 5]), text="test")
        except ValueError:
            pass

    def test_rw_internal(self):
        pb = self.utt.write_internal()
        self.utt.read_internal(pb)

    def test_rw(self):
        with tempfile.TemporaryDirectory() as tpf:
            output_file = os.path.join(tpf, "test.data")
            self.utt.write(output_file)
            self.assertTrue(os.path.exists(output_file))
            self.utt.read(output_file)

    def test_rw_mat_empty(self):
        np_mat = np.array([])
        utterance.numpy_to_mat(np_mat, self.float_mat)
        self.assertEqual(self.float_mat.num_row, 0)
        self.assertEqual(self.float_mat.num_col, 0)
        np_mat_recover = utterance.mat_to_numpy(self.float_mat)
        self.assertEqual(np_mat_recover.shape, np_mat.shape)
        self.assertTrue((np_mat == np_mat_recover).all())

    def test_rw_mat_scalar(self):
        np_mat = np.array([1])
        utterance.numpy_to_mat(np_mat, self.float_mat)
        self.assertEqual(self.float_mat.num_row, 1)
        self.assertEqual(self.float_mat.num_col, 1)
        np_mat_recover = utterance.mat_to_numpy(self.float_mat)
        self.assertEqual(np_mat_recover.shape, np_mat.shape)
        self.assertTrue((np_mat == np_mat_recover).all())

    def test_rw_mat_row_vec(self):
        np_mat = np.array([1, 2, 3, 4])
        num_ele = len(np_mat)
        utterance.numpy_to_mat(np_mat, self.float_mat)
        self.assertEqual(self.float_mat.num_row, 1)
        self.assertEqual(self.float_mat.num_col, num_ele)
        np_mat_recover = utterance.mat_to_numpy(self.float_mat)
        self.assertEqual(np_mat_recover.shape, np_mat.shape)
        self.assertTrue((np_mat == np_mat_recover).all())

    def test_rw_mat_col_vec(self):
        np_mat = np.array([[1], [2], [3], [4]])
        num_row, num_col = np_mat.shape
        utterance.numpy_to_mat(np_mat, self.float_mat)
        self.assertEqual(self.float_mat.num_row, num_row)
        self.assertEqual(self.float_mat.num_col, num_col)
        np_mat_recover = utterance.mat_to_numpy(self.float_mat)
        self.assertEqual(np_mat_recover.shape, (num_row, num_col))
        self.assertTrue((np_mat == np_mat_recover).all())

    def test_rw_mat_int_mat(self):
        np_mat = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        np_mat.astype(int)
        num_row, num_col = np_mat.shape
        utterance.numpy_to_mat(np_mat, self.int_mat)
        self.assertEqual(self.int_mat.num_row, num_row)
        self.assertEqual(self.int_mat.num_col, num_col)
        np_mat_recover = utterance.mat_to_numpy(self.int_mat)
        self.assertEqual(np_mat_recover.shape, (num_row, num_col))
        self.assertTrue((np_mat == np_mat_recover).all())

    def test_rw_mat_float_mat(self):
        np_mat = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        num_row, num_col = np_mat.shape
        utterance.numpy_to_mat(np_mat, self.float_mat)
        self.assertEqual(self.float_mat.num_row, num_row)
        self.assertEqual(self.float_mat.num_col, num_col)
        np_mat_recover = utterance.mat_to_numpy(self.float_mat)
        self.assertEqual(np_mat_recover.shape, (num_row, num_col))
        self.assertTrue((np_mat == np_mat_recover).all())

    def test_rw_mat_binary_mat(self):
        np_mat = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
        np_mat.astype(bool)
        num_row, num_col = np_mat.shape
        utterance.numpy_to_mat(np_mat, self.bool_mat)
        self.assertEqual(self.bool_mat.num_row, num_row)
        self.assertEqual(self.bool_mat.num_col, num_col)
        np_mat_recover = utterance.mat_to_numpy(self.bool_mat)
        self.assertEqual(np_mat_recover.shape, (num_row, num_col))
        self.assertTrue((np_mat == np_mat_recover).all())

    def test_rw_segment(self):
        utterance.write_segment(self.tier, self.seg)
        self.assertEqual(self.seg.num_item, len(self.tier.intervals))
        interval = utterance.read_segment(self.seg)
        self.assertTrue(isinstance(interval, IntervalTier))
        self.assertEqual(len(self.tier.intervals), len(interval.intervals))

    def test_rw_data(self):
        data = self.utt.data
        self.utt.data = data

    def test_rw_wav(self):
        input_wav = np.array([1, 2, 3, 4])
        self.utt.wav = input_wav
        wav = self.utt.wav
        self.assertTrue((input_wav == wav).any())

    def test_rw_fs(self):
        self.utt.fs = 16000
        fs = self.utt.fs
        self.assertEqual(fs, 16000)

    def test_set_invalid_fs(self):
        try:
            self.utt.fs = 0
        except ValueError:
            pass

        try:
            self.utt.fs = -2
        except ValueError:
            pass

    def test_rw_text(self):
        self.utt.text = "test"
        text = self.utt.text
        self.assertEqual(text, "test")

    def test_rw_align(self):
        align = TextGrid()
        align.read("data/test.TextGrid")
        self.utt.align = align
        align_recover = self.utt.align
        self.assertEqual(len(align_recover.tiers), len(align.tiers))

    def test_rw_ppg(self):
        ppg = np.random.uniform(0, 1, (100, 5816))
        self.utt.ppg = ppg
        ppg_recover = self.utt.ppg
        # There might be a small difference since the data are saved in float
        # while the original values are double precision
        self.assertAlmostEqual(((ppg_recover - ppg) ** 2).sum(), 0)

    def test_rw_monophone_ppg(self):
        mono_ppg = np.random.uniform(0, 1, (100, 40))
        self.utt.monophone_ppg = mono_ppg
        mono_ppg_recover = self.utt.monophone_ppg
        # There might be a small difference since the data are saved in float
        # while the original values are double precision
        self.assertAlmostEqual(((mono_ppg_recover - mono_ppg) ** 2).sum(), 0)

    def test_rw_phone(self):
        self.utt.phone = self.tier
        interval = self.utt.phone
        self.assertEqual(len(interval.intervals), len(self.tier.intervals))

    def test_rw_word(self):
        self.utt.word = self.tier
        interval = self.utt.word
        self.assertEqual(len(interval.intervals), len(self.tier.intervals))

    def test_rw_lab(self):
        lab = np.array([1, 2, 3, 4, 5, 5, 6, 7])
        self.utt.lab = lab
        lab_recover = self.utt.lab
        self.assertTrue((lab == lab_recover).all())

    def test_rw_utterance_id(self):
        self.utt.utterance_id = "id"
        utt_id = self.utt.utterance_id
        self.assertEqual(utt_id, "id")

    def test_rw_speaker_id(self):
        self.utt.speaker_id = "id"
        spk_id = self.utt.speaker_id
        self.assertEqual(spk_id, "id")

    def test_rw_dialect(self):
        self.utt.dialect = "EN_CN"
        dialect = self.utt.dialect
        self.assertEqual(dialect, "EN_CN")

    def test_rw_gender(self):
        self.utt.gender = "O"
        gender = self.utt.gender
        self.assertEqual(gender, "O")

    def test_rw_original_file(self):
        self.utt.original_file = "file.test"
        origin_file = self.utt.original_file
        self.assertEqual(origin_file, "file.test")

    def test_rw_num_channel(self):
        self.utt.num_channel = 2
        channel = self.utt.num_channel
        self.assertEqual(channel, 2)

    def test_rw_kaldi_shift(self):
        self.utt.kaldi_shift = 10
        shift = self.utt.kaldi_shift
        self.assertEqual(shift, 10)

    def test_rw_kaldi_window_size(self):
        self.utt.kaldi_window_size = 25
        w_size = self.utt.kaldi_window_size
        self.assertEqual(w_size, 25)

    def test_rw_kaldi_window_type(self):
        self.utt.kaldi_window_type = "hamming"
        w_type = self.utt.kaldi_window_type
        self.assertEqual(w_type, "hamming")

    def test_rw_vocoder(self):
        self.utt.vocoder = "WORLD"
        vocoder_name = self.utt.vocoder
        self.assertEqual(vocoder_name, "WORLD")

    def test_rw_spec(self):
        spec = np.random.uniform(0, 1, (100, 513))
        self.utt.spec = spec
        spec_recover = self.utt.spec
        # There might be a small difference since the data are saved in float
        # while the original values are double precision
        self.assertAlmostEqual(((spec_recover - spec) ** 2).sum(), 0)
        self.assertEqual(self.utt.spec_dim, 513)

    def test_rw_mfcc(self):
        mfcc = np.random.uniform(0, 1, (100, 25))
        self.utt.mfcc = mfcc
        mfcc_recover = self.utt.mfcc
        # There might be a small difference since the data are saved in float
        # while the original values are double precision
        self.assertAlmostEqual(((mfcc_recover - mfcc) ** 2).sum(), 0)
        self.assertEqual(self.utt.mfcc_dim, 25)

    def test_rw_mcep(self):
        mcep = np.random.uniform(0, 1, (100, 60))
        self.utt.mcep = mcep
        mcep_recover = self.utt.mcep
        # There might be a small difference since the data are saved in float
        # while the original values are double precision
        self.assertAlmostEqual(((mcep_recover - mcep) ** 2).sum(), 0)
        self.assertEqual(self.utt.mcep_dim, 60)

    def test_rw_f0(self):
        f0 = np.array([1, 2, 3, 4, 5])
        self.utt.f0 = f0
        f0_recover = self.utt.f0
        # There might be a small difference since the data are saved in float
        # while the original values are double precision
        self.assertAlmostEqual(((f0_recover - f0) ** 2).sum(), 0)
        self.assertEqual(self.utt.num_frame, 5)

    def test_rw_ap(self):
        ap = np.random.uniform(0, 1, (100, 513))
        self.utt.ap = ap
        ap_recover = self.utt.ap
        # There might be a small difference since the data are saved in float
        # while the original values are double precision
        self.assertAlmostEqual(((ap_recover - ap) ** 2).sum(), 0)
        self.assertEqual(self.utt.ap_dim, 513)

    def test_rw_bap(self):
        bap = np.array([1, 2, 3, 4, 5])
        self.utt.bap = bap
        bap_recover = self.utt.bap
        # There might be a small difference since the data are saved in float
        # while the original values are double precision
        self.assertAlmostEqual(((bap_recover - bap) ** 2).sum(), 0)
        self.assertEqual(self.utt.bap_dim, 1)

        bap = np.random.uniform(0, 1, (100, 5))
        self.utt.bap = bap
        bap_recover = self.utt.bap
        # There might be a small difference since the data are saved in float
        # while the original values are double precision
        self.assertAlmostEqual(((bap_recover - bap) ** 2).sum(), 0)
        self.assertEqual(self.utt.bap_dim, 5)

    def test_rw_vuv(self):
        vuv = np.array([1, 2, 3, 4, 5])
        self.utt.vuv = vuv
        vuv_recover = self.utt.vuv
        # There might be a small difference since the data are saved in float
        # while the original values are double precision
        self.assertAlmostEqual(((vuv_recover - vuv) ** 2).sum(), 0)

    def test_rw_temporal_position(self):
        tp = np.array([1, 2, 3, 4, 5])
        self.utt.temporal_position = tp
        tp_recover = self.utt.temporal_position
        # There might be a small difference since the data are saved in float
        # while the original values are double precision
        self.assertAlmostEqual(((tp_recover - tp) ** 2).sum(), 0)

    def test_rw_vocoder_shift(self):
        self.utt.vocoder_shift = 10
        shift = self.utt.vocoder_shift
        self.assertEqual(shift, 10)

    def test_rw_vocoder_window_size(self):
        self.utt.vocoder_window_size = 25
        w_size = self.utt.vocoder_window_size
        self.assertEqual(w_size, 25)

    def test_rw_vocoder_window_type(self):
        self.utt.vocoder_window_type = "hamming"
        w_type = self.utt.vocoder_window_type
        self.assertEqual(w_type, "hamming")

    def test_rw_num_frame(self):
        self.utt.num_frame = 10
        num_frame = self.utt.num_frame
        self.assertEqual(num_frame, 10)

    def test_rw_alpha(self):
        self.utt.alpha = 0.42
        alpha = self.utt.alpha
        self.assertAlmostEqual(alpha, 0.42)

    def test_rw_fft_size(self):
        self.utt.fft_size = 1024
        fft_size = self.utt.fft_size
        self.assertEqual(fft_size, 1024)

    def test_rw_spec_dim(self):
        self.utt.spec_dim = 513
        spec_dim = self.utt.spec_dim
        self.assertEqual(spec_dim, 513)

    def test_rw_mfcc_dim(self):
        self.utt.mfcc_dim = 60
        mfcc_dim = self.utt.mfcc_dim
        self.assertEqual(mfcc_dim, 60)

    def test_rw_mcep_dim(self):
        self.utt.mcep_dim = 60
        mcep_dim = self.utt.mcep_dim
        self.assertEqual(mcep_dim, 60)

    def test_rw_f0_floor(self):
        self.utt.f0_floor = 40
        f0_floor = self.utt.f0_floor
        self.assertEqual(f0_floor, 40)

    def test_rw_f0_ceil(self):
        self.utt.f0_ceil = 800
        f0_ceil = self.utt.f0_ceil
        self.assertEqual(f0_ceil, 800)

    def test_rw_timestamp(self):
        self.utt.timestamp = '1/25/2019'
        timestamp = self.utt.timestamp
        self.assertEqual(timestamp, '1/25/2019')

    def test_rw_ap_dim(self):
        self.utt.ap_dim = 5
        ap_dim = self.utt.ap_dim
        self.assertEqual(ap_dim, 5)

    def test_rw_bap_dim(self):
        self.utt.bap_dim = 1
        bap_dim = self.utt.bap_dim
        self.assertEqual(bap_dim, 1)

    def test_rw_pitch_tracker(self):
        self.utt.pitch_tracker = 'dio'
        pitch_tracker = self.utt.pitch_tracker
        self.assertEqual(pitch_tracker, 'dio')

    def test_time_to_frame_valid_input(self):
        self.assertEqual(utterance.time_to_frame(3.66, 5), 732)
        self.assertEqual(utterance.time_to_frame(0, 5), 0)

    def test_time_to_frame_invalid_input(self):
        try:
            utterance.time_to_frame(-10, 5)
        except ValueError:
            pass

    def test_time_to_frame_interval_tier_simple_case(self):
        shift = 5
        frame_tier = utterance.time_to_frame_interval_tier(self.tier, shift)
        self.assertEqual(frame_tier.minTime, 0)
        self.assertEqual(frame_tier.maxTime, utterance.time_to_frame(
            self.tier.maxTime, shift))

    def test_time_to_frame_interval_tier_short_seg(self):
        tier = IntervalTier('test', 0, 0.01)
        tier.add(0, 0.003, "a")
        tier.add(0.003, 0.01, "b")
        frame_tier = utterance.time_to_frame_interval_tier(tier, 5)
        self.assertEqual(frame_tier.minTime, 0)
        self.assertEqual(frame_tier.maxTime, 2)
        self.assertEqual(frame_tier.intervals[0].minTime, 0)
        self.assertEqual(frame_tier.intervals[0].maxTime, 1)
        self.assertEqual(frame_tier.intervals[1].minTime, 1)
        self.assertEqual(frame_tier.intervals[1].maxTime, 2)

    def test_is_sil(self):
        self.assertTrue(utterance.is_sil("sil"))
        self.assertTrue(utterance.is_sil("sp"))
        self.assertTrue(utterance.is_sil("spn"))
        self.assertTrue(utterance.is_sil(""))
        self.assertTrue(utterance.is_sil("SIL"))
        self.assertTrue(utterance.is_sil("SPN"))
        self.assertTrue(utterance.is_sil("SP"))
        self.assertFalse(utterance.is_sil("AO"))
        self.assertFalse(utterance.is_sil("s"))

    def test_normalize_phone(self):
        self.assertEqual("ao", utterance.normalize_phone("AO0"))
        self.assertEqual("ao", utterance.normalize_phone("AO"))
        self.assertEqual("ao", utterance.normalize_phone("AO0, AH, s"))
        self.assertEqual("ao,ah,s",
                         utterance.normalize_phone("AO0, AH, s", False))
        self.assertEqual("sil", utterance.normalize_phone("SIL"))
        self.assertEqual("sil", utterance.normalize_phone(""))
        try:
            utterance.normalize_phone("1243")
        except ValueError:
            pass

    def test_normalize_tier_mark(self):
        utterance.normalize_tier_mark(self.tier)
        utterance.normalize_tier_mark(self.tier, "NormalizePhoneAnnotation")

        try:
            utterance.normalize_tier_mark(self.tier, "NormalizePhone")
        except ValueError:
            pass

    def test_read_sym_table(self):
        sym_table = utterance.read_sym_table("data/phoneme_table")
        self.assertEqual(len(sym_table), 40)

    def test_get_hardcoded_sym_table(self):
        sym_table = utterance.get_hardcoded_sym_table()
        correct_sym_table = utterance.read_sym_table("data/phoneme_table")
        self.assertTrue(
            set(sym_table.items()) == set(correct_sym_table.items()))

    def test_get_phone_tier_invalid_0(self):
        try:
            self.utt.get_phone_tier()
        except ValueError:
            pass

    def test_get_phone_tier_invalid_1(self):
        try:
            self.utt.kaldi_shift = 5
            self.utt.get_phone_tier()
        except ValueError:
            pass

    def test_get_word_tier_invalid_0(self):
        try:
            self.utt.get_word_tier()
        except ValueError:
            pass

    def test_get_word_tier_invalid_1(self):
        try:
            self.utt.kaldi_shift = 5
            self.utt.get_word_tier()
        except ValueError:
            pass

    def test_get_monophone_ppg_valid(self):
        fs, wav = wavfile.read('data/test_mono_channel.wav')
        utt = utterance.Utterance(wav, fs)
        utt.kaldi_shift = 5
        ppgs = utt.get_monophone_ppg()
        self.assertEqual(ppgs.shape[1], 40)

    def test_get_monophone_ppg_invalid_0(self):
        try:
            self.utt.get_monophone_ppg()
        except ValueError:
            pass

    def test_get_monophone_ppg_invalid_1(self):
        try:
            self.utt.kaldi_shift = 5
            self.utt.get_monophone_ppg()
        except ValueError:
            pass
