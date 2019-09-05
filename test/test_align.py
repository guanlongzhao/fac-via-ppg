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
from common import align
from textgrid import TextGrid


class TestAligner(unittest.TestCase):
    def setUp(self):
        self.tg_file = "data/test.TextGrid"

    def tearDown(self):
        pass

    def test_write_tg_to_str(self):
        tg = TextGrid()
        tg.read(self.tg_file, 2)
        tg_str = align.write_tg_to_str(tg)
        self.assertTrue(tg_str.startswith('File type = "ooTextFile"'))

    def test_read_tg_from_str(self):
        with open(self.tg_file, 'r') as reader:
            tg_str = reader.read()
        tg = align.read_tg_from_str(tg_str)
        self.assertTrue(isinstance(tg, TextGrid))
        self.assertEqual(len(tg.tiers), 2)
