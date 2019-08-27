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

import os
import tempfile
import unittest
from common import align
from glob import glob
from scipy.io import wavfile
from textgrid import TextGrid


class TestAligner(unittest.TestCase):
    def setUp(self):
        self.wav_path = "data/align/batch/test_01.wav"
        self.text_path = "data/align/batch/test_01.lab"
        self.fs, self.wav = wavfile.read(self.wav_path, False)
        with open(self.text_path, 'r') as reader:
            self.text = reader.readline()
        self.aligner = align.MontrealAligner(
            am_path="data/align/test_english_am.zip",
            dict_path="data/align/test_dict")

    def tearDown(self):
        pass

    def test_align_single_internal_mono(self):
        tg = self.aligner.align_single_internal(self.wav, self.fs, self.text)
        self.assertEqual(len(tg.tiers), 2)

    def test_align_single_internal_dual(self):
        fs, wav = wavfile.read("data/test_dual_channel.wav", False)
        with open("data/test_dual_channel.txt", 'r') as reader:
            text = reader.readline()
        tg = self.aligner.align_single_internal(wav, fs, text)
        self.assertEqual(len(tg.tiers), 2)

    def test_align_single_wf_tf(self):
        # wav is file, text is file
        tg = self.aligner.align_single(self.wav_path, self.text_path)
        self.assertEqual(len(tg.tiers), 2)

    def test_align_single_wd_tf(self):
        # wav is data, text is file
        tg = self.aligner.align_single(self.wav, self.text_path, self.fs,
                                       is_wav_path=False)
        self.assertEqual(len(tg.tiers), 2)

    def test_align_single_wf_td(self):
        # wav is file, text is data
        tg = self.aligner.align_single(self.wav_path, self.text,
                                       is_text_path=False)
        self.assertEqual(len(tg.tiers), 2)

    def test_align_single_wd_td(self):
        # wav is data, text is data
        tg = self.aligner.align_single(self.wav, self.text, self.fs, False,
                                       False)
        self.assertEqual(len(tg.tiers), 2)

    def test_align_batch(self):
        input_dir = "data/align/batch"
        with tempfile.TemporaryDirectory() as output_dir:
            actual_output_dir = self.aligner.align_batch(input_dir, output_dir)
            tgs = glob(os.path.join(actual_output_dir, "*.TextGrid"))
            num_tgs = len(tgs)
            num_wavs = len(glob(os.path.join(input_dir, "*.wav")))
            # Check each TG file to make sure it's valid
            for each_tg in tgs:
                tg = TextGrid()
                tg.read(each_tg)
                self.assertEqual(len(tg.tiers), 2)
            self.assertEqual(num_tgs, num_wavs)
            self.assertEqual(actual_output_dir, output_dir)

    def test_write_tg_to_str(self):
        tg_file = "data/test.TextGrid"
        tg = TextGrid()
        tg.read(tg_file, 2)
        tg_str = align.write_tg_to_str(tg)
        self.assertTrue(tg_str.startswith('File type = "ooTextFile"'))

    def test_read_tg_from_str(self):
        tg_file = "data/test.TextGrid"
        with open(tg_file, 'r') as reader:
            tg_str = reader.read()
        tg = align.read_tg_from_str(tg_str)
        self.assertTrue(isinstance(tg, TextGrid))
        self.assertEqual(len(tg.tiers), 2)
