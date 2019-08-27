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

"""This script processes data utterances for the accent conversion task."""

import logging
import os
import time
from common.feat import read_wav_kaldi_internal, compute_mfcc_plus_ivector
from common.utterance import Utterance
from glob import glob
from ppg import DependenciesPPG, compute_monophone_ppg, \
    compute_full_ppg_wrapper, compute_chain_full_ppg
from scipy.io import wavfile
from textgrid import TextGrid
from common import decode
from kaldi.matrix import Vector


CHAIN_AM_PATH = '/home/guanlong/PycharmProjects/ppg-speech/data/am/lstm/' \
                'final.mdl'
CHAIN_DEN_FST_PATH = '/home/guanlong/PycharmProjects/ppg-speech/data/am/lstm' \
                     '/den.fst'
CHAIN_FEAT_CONFIG_PATH = '/home/guanlong/PycharmProjects/ppg-speech/data/' \
                         'feats/chain_configs/online.conf'


def process_utt(wav, fs, text, tg, ppg_deps, is_vocoder=False,
                is_mono_ppg=False, is_full_ppg=True, speaker_code=None):
    """Generate data utterance for one sentence.

    Args:
        wav: Speech data in ndarray.
        fs: Sampling frequency.
        text: Transcript.
        tg: Textgrid object.
        ppg_deps: Dependencies for computing the PPGs.
        is_vocoder: If True, will compute vocoder features.
        is_mono_ppg: If True, will compute monophone PPGs.
        is_full_ppg: If True, will compute the full PPGs.
        speaker_code: If we have fmllr for the speaker, will use that transform.

    Returns:
        utt: The data utterance.
    """
    utt = Utterance(wav, fs, text)
    utt.align = tg
    utt.kaldi_shift = 10  # ms
    wav_kaldi = read_wav_kaldi_internal(wav, fs)

    fmllr = None
    if speaker_code in ppg_deps.fmllr_mats:
        print('Using %s fMLLR.' % speaker_code)
        fmllr = ppg_deps.fmllr_mats[speaker_code]

    if is_mono_ppg:
        utt.monophone_ppg = compute_monophone_ppg(wav_kaldi, ppg_deps.nnet,
                                                  ppg_deps.lda,
                                                  ppg_deps.monophone_trans,
                                                  utt.kaldi_shift, fmllr=fmllr)
    utt.get_phone_tier()
    utt.get_word_tier()

    if is_vocoder:
        utt.get_vocoder_feat()

    if is_full_ppg:
        utt.ppg = compute_full_ppg_wrapper(wav_kaldi, ppg_deps.nnet,
                                           ppg_deps.lda, utt.kaldi_shift,
                                           fmllr=fmllr)
    return utt


def process_utt_chain_model_with_ref(
        wav, wav_resample, fs, fs_resample,  text, tg, nnet, den_fst,
        feat_config, ref_wav, is_vocoder=False, is_full_ppg=True):
    """Generate data utterance for one sentence.

    Args:
        wav: Speech data in ndarray.
        fs: Sampling frequency.
        text: Transcript.
        tg: Textgrid object.
        is_vocoder: If True, will compute vocoder features.
        is_full_ppg: If True, will compute the full PPGs.

    Returns:
        utt: The data utterance.
    """
    utt = Utterance(wav, fs, text)
    utt.align = tg
    utt.kaldi_shift = 10  # ms
    wav_kaldi = read_wav_kaldi_internal(wav_resample, fs_resample)
    ref_wav_kaldi = read_wav_kaldi_internal(ref_wav, fs_resample)

    utt.get_phone_tier()
    utt.get_word_tier()

    if is_vocoder:
        utt.get_vocoder_feat()

    if is_full_ppg:
        _, ref_ivec = compute_mfcc_plus_ivector(ref_wav_kaldi, feat_config)
        # Copy ivector from the reference.
        ref_ivec_single = Vector(ref_ivec.num_cols).copy_row_from_mat_(
            ref_ivec, 0)
        mfcc, ivec = compute_mfcc_plus_ivector(wav_kaldi, feat_config)
        print('Copy the ref ivector...')
        for ii in range(ref_ivec.num_rows):
            ivec.copy_row_from_vec_(ref_ivec_single, ii)
        last_ivec_single = Vector(ref_ivec.num_cols).copy_row_from_mat_(
            ref_ivec, ref_ivec.num_rows - 1)
        ivec.copy_row_from_vec_(last_ivec_single, ref_ivec.num_rows - 1)
        utt.ppg = compute_chain_full_ppg(nnet, den_fst, mfcc, ivec)
    return utt


def process_utt_chain_model(wav, wav_resample, fs, fs_resample, text, tg,
                            nnet, den_fst, feat_config, is_vocoder=False,
                            is_full_ppg=True):
    """Generate data utterance for one sentence.

    Args:
        wav: Speech data in ndarray.
        fs: Sampling frequency.
        text: Transcript.
        tg: Textgrid object.
        is_vocoder: If True, will compute vocoder features.
        is_full_ppg: If True, will compute the full PPGs.

    Returns:
        utt: The data utterance.
    """
    utt = Utterance(wav, fs, text)
    utt.align = tg
    utt.kaldi_shift = 10  # ms
    wav_kaldi = read_wav_kaldi_internal(wav_resample, fs_resample)

    utt.get_phone_tier()
    utt.get_word_tier()

    if is_vocoder:
        utt.get_vocoder_feat()

    if is_full_ppg:
        mfcc, ivec = compute_mfcc_plus_ivector(wav_kaldi, feat_config)
        utt.ppg = compute_chain_full_ppg(nnet, den_fst, mfcc, ivec)
    return utt


def process_speaker(path, gender='', dialect='', skip_exist=True,
                    is_chain=False, ref_speaker_wav_path=None):
    """Process data for one speaker.

    Args:
        path: Root dir to that speaker.
        gender: [optional] Gender.
        dialect: [optional] Dialect.
        skip_exist: [optional] If True then skip files that have already cached.
        ref_speaker_wav_path: should be the downsampled wav path to the ref.
    """
    if not os.path.isdir(path):
        logging.error('Path %s does not exist.', path)
    speaker_id = os.path.basename(path)
    wav_dir = os.path.join(path, 'wav')
    wav_dir_8k = os.path.join(path, 'wav_8k')
    if not os.path.isdir(wav_dir):
        logging.error('Path %s does not exist.', wav_dir)
    tg_dir = os.path.join(path, 'tg')
    if not os.path.isdir(tg_dir):
        logging.error('Path %s does not exist.', tg_dir)
    text_dir = os.path.join(path, 'text')
    if not os.path.isdir(text_dir):
        logging.error('Path %s does not exist.', text_dir)
    cache_dir = os.path.join(path, 'cache')
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    ppg_deps = DependenciesPPG()
    print(ppg_deps.nnet_path)
    if is_chain:
        nnet = decode.read_nnet3_model(CHAIN_AM_PATH)
        den_fst = decode.read_den_fst(CHAIN_DEN_FST_PATH)
    for wav_file in glob(os.path.join(wav_dir, '*.wav')):
        basename = os.path.basename(wav_file).split('.')[0]
        cache_file = os.path.join(cache_dir, basename + '.proto')
        if skip_exist and os.path.isfile(cache_file):
            logging.info('Skip existing cache.')
            continue
        utt_id = '%s_%s' % (speaker_id, basename)
        logging.info('Processing utterance %s', utt_id)
        fs, wav = wavfile.read(wav_file, False)
        tg_file = os.path.join(tg_dir, basename + '.TextGrid')
        tg = TextGrid()
        tg.read(tg_file)
        text_file = os.path.join(text_dir, basename + '.lab')
        with open(text_file, 'r') as reader:
            text = reader.readline()
        start = time.time()
        if is_chain:
            wav_file_8k = os.path.join(wav_dir_8k, basename + '.wav')
            fs_8k, wav_8k = wavfile.read(wav_file_8k, False)

            if ref_speaker_wav_path is None:
                utt = process_utt_chain_model(
                    wav, wav_8k, fs, fs_8k, text, tg,
                    nnet, den_fst, CHAIN_FEAT_CONFIG_PATH)
            else:
                ref_wav_file_8k = os.path.join(ref_speaker_wav_path,
                                               basename + '.wav')
                ref_fs_8k, ref_wav_8k = wavfile.read(ref_wav_file_8k, False)
                assert ref_fs_8k == fs_8k
                utt = process_utt_chain_model_with_ref(
                    wav, wav_8k, fs, fs_8k, text, tg,
                    nnet, den_fst, CHAIN_FEAT_CONFIG_PATH, ref_wav_8k)
        else:
            utt = process_utt(wav, fs, text, tg, ppg_deps,
                              speaker_code=speaker_id)
        end = time.time()
        logging.info('Took %1.2f second(s).', end - start)
        utt.speaker_id = speaker_id
        utt.utterance_id = utt_id
        utt.original_file = wav_file
        if len(dialect) > 0:
            utt.dialect = dialect
        if len(gender) > 0:
            utt.gender = gender
        utt.write(cache_file)


def main(path, skip_exist=True):
    """Process all speakers.

    Args:
        path: Root dir to all speaker data and the metadata file.
        skip_exist: [optional] If True then skip files that have already cached.
    """
    meta_data_file = os.path.join(path, 'metadata')
    with open(meta_data_file, 'r') as reader:
        for ii, line in enumerate(reader):
            if ii == 0:
                # Skip the header
                continue
            name, gender, dialect = line.split()
            curr_path = os.path.join(path, name)
            process_speaker(curr_path, gender, dialect, skip_exist)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    root_dir = '/media/guanlong/DATA1/exp/ppg-speech/data'
    start = time.time()
    main(root_dir, True)
    end = time.time()
    logging.info('All the steps took %f seconds.', end - start)
