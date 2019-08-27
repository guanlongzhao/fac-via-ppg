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
from common import decode


class TestDecodeFunctions(unittest.TestCase):
    def setUp(self):
        self.nnet_path = "data/nnet3.raw"

    def tearDown(self):
        pass

    def test_read_nnet3_modol(self):
        nnet = decode.read_nnet3_model(self.nnet_path)
        self.assertEqual(nnet.input_dim("input"), 40)
