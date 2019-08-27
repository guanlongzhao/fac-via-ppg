# Copyright 2019 Guanlong Zhao

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script re-processes data utterances to fix the vocoder features."""

import logging
import os
import time
from common.utterance import Utterance
from glob import glob


def process_speaker(path):
    """Process data for one speaker.

    Args:
        path: Root dir to that speaker.
    """
    if not os.path.isdir(path):
        logging.error('Path %s does not exist.', path)
    speaker_id = os.path.basename(path)
    cache_dir = os.path.join(path, 'cache')
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    for cache_file in glob(os.path.join(cache_dir, '*.proto')):
        basename = os.path.basename(cache_file).split('.')[0]
        utt_id = '%s_%s' % (speaker_id, basename)
        logging.info('Processing utterance %s', utt_id)
        curr_start = time.time()
        utt = Utterance()
        utt.read(cache_file)
        utt.get_vocoder_feat()
        curr_end = time.time()
        logging.info('Took %1.2f second(s).', curr_end - curr_start)
        utt.write(cache_file)


def main(path):
    """Process all speakers.

    Args:
        path: Root dir to all speaker data and the metadata file.
    """
    meta_data_file = os.path.join(path, 'metadata')
    with open(meta_data_file, 'r') as reader:
        for ii, line in enumerate(reader):
            if ii == 0:
                # Skip the header
                continue
            name, gender, dialect = line.split()
            curr_path = os.path.join(path, name)
            process_speaker(curr_path)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    root_dir = '/data_repo/arctic'
    start = time.time()
    main(root_dir)
    end = time.time()
    logging.info('All the steps took %f seconds.', end - start)
