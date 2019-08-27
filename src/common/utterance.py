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

"""This module provides a base data utterance class and helper functions."""

import datetime
import logging
import numpy as np
import ppg
import pyworld as pw
import re
import time
import math
from common.data_utterance_pb2 import DataUtterance, Segment, MetaData,\
    VocoderFeature
from common.align import write_tg_to_str, read_tg_from_str, MontrealAligner
from common.feat import read_wav_kaldi_internal
from numpy import ndarray
from scipy.io import wavfile
from textgrid import TextGrid, IntervalTier


# 48Hz is the minimum for a fft_size of 1024 for fs=16KHz. The value is defined
# by 3*fs/(fft_size-3)
DEFAULT_F0_FLOOR = 48  # Hz
# Titze, I.R. (1994). Principles of Voice Production, Prentice Hall (currently
# published by NCVS.org) (pp. 188)
DEFAULT_F0_CEIL = 400  # Hz
DEFAULT_SHIFT = 5  # ms
DEFAULT_PITCH_TRACKER = 'harvest'  # 'harvest' | 'dio'
DEFAULT_FFT_SIZE = 1024  # 513-dim spec
DEFAULT_MCEP_DIM = 60  # From Merlin default


def mat_to_numpy(mat):
    """Convert a matrix protobuffer message to a numpy ndarray.

    The matrix message can be any matrix defined in data_utterance.pb.

    Args:
        mat: A matrix message.

    Returns:
        mat_numpy: Matrix saved in ndarray format.
    """
    num_row = mat.num_row
    num_col = mat.num_col
    flat_mat = np.array(mat.data)
    if num_row > 1:
        mat_numpy = flat_mat.reshape((num_row, num_col))
    else:
        # Due to numpy stupidity, why are they treating a row vector different
        # than everything else? WTF?!
        mat_numpy = flat_mat.reshape(num_col)
    return mat_numpy


def numpy_to_mat(np_mat, mat):
    """Save a numpy ndarray to a matrix message in place.

    The existing content in the old matrix will be erased.
    The matrix message can be any matrix defined in data_utterance.pb.

    Args:
        np_mat: A matrix in ndarray format.
        mat: A matrix message.
    """
    mat.Clear()
    dims = np_mat.shape  # rows*cols, might be one dim (row vector)
    mat.data.extend(np_mat.flatten())  # To 1-D array
    if np_mat.size > 0:
        # Non empty matrix
        if len(dims) > 1:
            # 2-D matrix
            mat.num_row = dims[0]
            mat.num_col = dims[1]
        else:
            # 1-D matrix, in this case, numpy only return one shape dim
            # for row vectors -> they become (D,). Therefore, the number of
            # rows should be 1, and number of cols should be D.
            mat.num_row = 1
            mat.num_col = dims[0]
    else:
        # Empty matrix
        mat.num_row = 0
        mat.num_col = 0


def read_segment(val: Segment) -> IntervalTier:
    """Read a Segment message and save it to an IntervalTier object.

    Args:
        val: A Segment message as defined in data_utterance.pb.

    Returns:
        interval: The Segment message saved in an IntervalTier object.
    """
    symbols = val.symbol
    start_time = mat_to_numpy(val.start_time)
    end_time = mat_to_numpy(val.end_time)
    num_items = val.num_item

    if not (len(symbols) == len(start_time) == len(end_time) == num_items):
        raise ValueError("Interval item number is not consistent!")

    interval = IntervalTier(minTime=start_time[0], maxTime=end_time[-1])
    for sym, min_time, max_time in zip(symbols, start_time, end_time):
        interval.add(min_time, max_time, sym)
    return interval


def write_segment(val: IntervalTier, seg: Segment):
    """Write an IntervalTier object to a Segment message in place.

    Args:
        val: An IntervalTier object.
        seg: A Segment message as defined in data_utterance.pb.
    """
    seg.Clear()
    num_item = len(val.intervals)
    start_time = []
    end_time = []
    for each_interval in val.intervals:
        seg.symbol.append(each_interval.mark)
        start_time.append(each_interval.minTime)
        end_time.append(each_interval.maxTime)
    numpy_to_mat(np.array(start_time), seg.start_time)
    numpy_to_mat(np.array(end_time), seg.end_time)
    seg.num_item = num_item


def time_to_frame(t, shift):
    """Convert time (in seconds) to frame index (zero-indexed).

    The frame index means that this timestamp belongs to this frame.

    Args:
        t: Time in seconds. This marks the start time of the frame.
        shift: Window shift in ms.

    Returns:
        frame_idx: Frame index (non-negative int).
    """
    if t < 0:
        raise ValueError("Time should be positive!")
    frame_idx = math.floor(
        float(t) * 1000 / float(shift))  # The frame this timestamp belongs to.
    frame_idx = int(frame_idx)
    assert frame_idx >= 0, "Frame index should be non-negative."
    return frame_idx


def time_to_frame_interval_tier(time_tier: IntervalTier,
                                shift) -> IntervalTier:
    """Convert an IntervalTier in time to frame.

    Args:
        time_tier: IntervalTier represented in seconds.
        shift: Window shift in ms.

    Returns:
        frame_tier: IntervalTier represented in frames.
    """
    max_frame = time_to_frame(time_tier.maxTime, shift)
    frame_tier = IntervalTier(time_tier.name, 0, max_frame)

    # Deal with (occasionally) very short segments -- less than a frame shift
    # If we have consecutive very small segments then the function will raise a
    # ValueError
    start_shift = 0
    for each_interval in time_tier.intervals:
        curr_min_frame = time_to_frame(each_interval.minTime, shift)
        if start_shift > 0:
            logging.warning("Last segment is too short, have to cut the %d "
                            "frame(s) from the beginning of the current "
                            "segment.", start_shift)
            curr_min_frame += start_shift
            start_shift = 0
        curr_max_frame = time_to_frame(each_interval.maxTime, shift)
        if curr_min_frame >= curr_max_frame:
            curr_max_frame = curr_min_frame + 1
            start_shift = curr_max_frame - curr_min_frame
            logging.warning("The current segment is too short, extend it for "
                            "%d frame(s).", start_shift)
        if curr_max_frame > frame_tier.maxTime:
            raise ValueError("Extreme short segments in the tier, please fix "
                             "these.")
        frame_tier.add(curr_min_frame, curr_max_frame, each_interval.mark)
    return frame_tier


def is_sil(s: str) -> bool:
    """Test if the input string represents silence.

    Args:
        s: A phoneme label.

    Returns:
        True if is silence, otherwise False.
    """
    if s.lower() in {"sil", "sp", "spn", ""}:
        return True
    else:
        return False


def normalize_phone(s: str, is_rm_annotation=True) -> str:
    """Normalize phoneme labels to lower case, stress-free form.

    This will also deal with L2-ARCTIC annotations.

    Args:
        s: A phoneme annotation.
        is_rm_annotation: [optional] Only return the canonical pronunciation if
        set to true, otherwise will keep the annotations.

    Returns:
        Normalized phoneme (canonical pronunciation or with annotations).
    """
    t = s.lower()
    pattern = re.compile(r"[^a-z,]")
    parse_tag = pattern.sub("", t)
    if is_sil(parse_tag):
        return "sil"
    if len(parse_tag) == 0:
        raise ValueError("Input %s is invalid.", s)
    if is_rm_annotation:
        # This handles the L2-ARCTIC annotations, here we extract the canonical
        # pronunciation
        return parse_tag.split(",")[0]
    else:
        return parse_tag


def normalize_word(s: str) -> str:
    """Normalize a word.

    Args:
        s: A word.
    Returns:
        The word in lower case.
    """
    return s.lower()


def normalize_tier_mark(tier: IntervalTier,
                        mode="NormalizePhoneCanonical") -> IntervalTier:
    """Normalize the marks of an IntervalTier.

    Refer to the code for supported modes.

    Args:
        tier: An IntervalTier object.
        mode: The filter function for each mark in the tier.

    Returns:
        tier: Mark-normalized tier.
    """
    if mode not in {"NormalizePhoneCanonical",
                    "NormalizePhoneAnnotation",
                    "NormalizeWord"}:
        raise ValueError("Mode %s is not valid.", mode)
    for each_interval in tier.intervals:
        if mode is "NormalizePhoneCanonical":
            # Only keep the canonical pronunciation.
            each_interval.mark = normalize_phone(each_interval.mark, True)
        elif mode is "NormalizePhoneAnnotation":
            # Keep the annotations.
            each_interval.mark = normalize_phone(each_interval.mark, False)
        elif mode is "NormalizeWord":
            each_interval.mark = normalize_word(each_interval.mark)
    return tier


def read_sym_table(sym_table_path: str) -> dict:
    """Read in a kaldi style symbol table.

    Each line in the symbol table file is "sym  index", separated by a
    whitespace.

    Args:
        sym_table_path: Path to the symbol table file.

    Returns:
        sym_table: A dictionary whose keys are symbols and values are indices.
    """
    sym_table = {}
    with open(sym_table_path, 'r') as reader:
        for each_line in reader:
            key, val = each_line.split()
            val = int(val)
            if key not in sym_table:
                sym_table[key] = val
            else:
                raise ValueError("Duplicated key: %s", key)
    return sym_table


def get_hardcoded_sym_table() -> dict:
    """Return the ARPABET phoneme symbol table.

    The dictionary is hard-coded so it is fast.
    """
    sym_table = {'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ay': 5, 'b': 6,
                 'ch': 7, 'd': 8, 'dh': 9, 'eh': 10, 'er': 11, 'ey': 12,
                 'f': 13, 'g': 14, 'hh': 15, 'ih': 16, 'iy': 17, 'jh': 18,
                 'k': 19, 'l': 20, 'm': 21, 'n': 22, 'ng': 23, 'ow': 24,
                 'oy': 25, 'p': 26, 'r': 27, 's': 28, 'sh': 29, 't': 30,
                 'th': 31, 'uh': 32, 'uw': 33, 'v': 34, 'w': 35, 'y': 36,
                 'z': 37, 'zh': 38, 'sil': 39}
    return sym_table


class Utterance(object):
    """Wrapper class for the protocol buffer data_utterance.

    Provides easy-to-use setters and getters to the protobuffer fields.
    """

    def __init__(self, wav=None, fs=-1, text=''):
        """Set necessary fields of an utterance.

        Args:
            wav: [optional] ndarray, a S*C matrix, S is number of samples and C
            is the number of channels.
            fs: [optional] Sampling frequency.
            text: [optional] Text transcription.
        """
        self._data = DataUtterance()
        if wav is None:
            wav = np.array([])
        if wav.size > 0 > fs:
            raise ValueError("Sampling frequency is not set!")
        self.wav = wav
        self.fs = fs
        self.text = text

    def read_internal(self, pb):
        """Read a DataUtterance from a raw string.

        Args:
            pb: A string containing a DataUtterance.
        """
        self._data.ParseFromString(pb)

    def read(self, pb_path):
        """Read a DataUtterance from a file.

        Args:
            pb_path: Path to a DataUtterance protobuffer file.
        """
        with open(pb_path, 'rb') as reader:
            self.read_internal(reader.read())

    def write_internal(self):
        """Write the DataUtterance to a string.

        Returns:
            Serialized DataUtterance string.
        """
        return self._data.SerializeToString()

    def write(self, pb_path):
        """Write a DataUtterance to a file.

        Args:
            pb_path: Path to a DataUtterance protobuffer file.
        """
        with open(pb_path, 'wb') as writer:
            writer.write(self.write_internal())

    def get_alignment(self) -> TextGrid:
        """A wrapper function to initialize the alignment of this utterance.

        Requires non-empty waveform, fs, and text data.

        Returns:
            tg: The alignment in TextGrid format.
        """
        if self.wav.size == 0 or self.fs < 0 or self.text == '':
            raise ValueError('To perform alignment, the object must contain '
                             'valid speech data, sampling frequency, and the '
                             'text transcript.')

        aligner = MontrealAligner()
        tg = aligner.align_single_internal(self.wav, self.fs, self.text)
        self.align = tg
        return tg

    def get_phone_tier(self) -> IntervalTier:
        """A wrapper function to initialize the phone tier of this utterance.

        Should only be called after obtained alignment and set kaldi_shift.

        Returns:
            phone_tier: The phone tier whose marks have been normalized and the
            times are in frames.
        """
        if self.kaldi_shift < 1:  # ms
            raise ValueError('Invalid frame kaldi frame shift parameter %d.',
                             self.kaldi_shift)
        if len(self.align) == 0:
            raise ValueError('Empty alignment, please run alignment first.')
        phone_tier = time_to_frame_interval_tier(self.align.getFirst('phones'),
                                                 self.kaldi_shift)
        phone_tier = normalize_tier_mark(phone_tier)
        self.phone = phone_tier
        return phone_tier

    def get_word_tier(self) -> IntervalTier:
        """A wrapper function to initialize the word tier of this utterance.

        Should only be called after obtained alignment and set kaldi_shift.

        Returns:
            word_tier: The word tier whose marks have been normalized and the
            times are in frames.
        """
        if self.kaldi_shift < 1:  # ms
            raise ValueError('Invalid frame kaldi frame shift parameter %d.',
                             self.kaldi_shift)
        if len(self.align) == 0:
            raise ValueError('Empty alignment, please run alignment first.')
        word_tier = time_to_frame_interval_tier(self.align.getFirst('words'),
                                                self.kaldi_shift)
        word_tier = normalize_tier_mark(word_tier, 'NormalizeWord')
        self.word = word_tier
        return word_tier

    def get_monophone_ppg(self) -> ndarray:
        """A wrapper function to initialize the monophone ppg of this utterance.

        Requires non-empty waveform, fs, and kaldi_shift.

        Returns:
            The monophone ppgs in numpy ndarray format.
        """
        if self.kaldi_shift < 1:  # ms
            raise ValueError('Invalid frame kaldi frame shift parameter %d.',
                             self.kaldi_shift)
        if self.wav.size == 0 or self.fs < 0:
            raise ValueError('To perform alignment, the object must contain '
                             'valid speech data and sampling frequency.')

        wav_kaldi = read_wav_kaldi_internal(self.wav, self.fs)
        ppg_deps = ppg.DependenciesPPG()
        self.monophone_ppg = ppg.compute_monophone_ppg(wav_kaldi, ppg_deps.nnet,
                                                       ppg_deps.lda,
                                                       ppg_deps.monophone_trans,
                                                       self.kaldi_shift)
        return self.monophone_ppg

    def get_vocoder_feat(self, **kwargs):
        """Extract vocoder features. Currently does not save the raw spec and ap

        Args:
            See 'options'.
        """
        options = {'f0_floor': DEFAULT_F0_FLOOR,
                   'f0_ceil': DEFAULT_F0_CEIL,
                   'shift': DEFAULT_SHIFT,
                   'pitch_tracker': DEFAULT_PITCH_TRACKER,
                   'fft_size': DEFAULT_FFT_SIZE,
                   'mcep_dim': DEFAULT_MCEP_DIM}
        for key, val in kwargs.items():
            if key in options:
                options[key] = val
            else:
                raise ValueError("Option %s not allowed!" % key)

        if self.wav.size == 0 or self.fs < 0:
            raise ValueError('To perform analysis, the object must contain '
                             'valid speech data and sampling frequency.')

        if self.wav.ndim >= 2:
            x = self.wav[:, 0]
        else:
            x = self.wav
        if x.max() > 1:  # If it is in the int16 format, convert to float.
            x = x / 32767  # Required by vocoder. [-1, 1].

        fs = self.fs

        # Extracting raw F0.
        if options['pitch_tracker'] == 'dio':
            _f0, t = pw.dio(x, fs, f0_floor=options['f0_floor'],
                            f0_ceil=options['f0_ceil'],
                            frame_period=options['shift'])
        elif options['pitch_tracker'] == 'harvest':
            _f0, t = pw.harvest(x, fs, f0_floor=options['f0_floor'],
                                f0_ceil=options['f0_ceil'],
                                frame_period=options['shift'])
        else:
            raise ValueError('F0 tracker %s not supported!' % options[
                'pitch_tracker'])
        # Refine F0.
        f0 = pw.stonemask(x, _f0, t, fs)
        # Spectrogram and mel-cepstrum.
        sp = pw.cheaptrick(x, f0, t, fs, f0_floor=options['f0_floor'],
                           fft_size=options['fft_size'])
        mcep = pw.code_spectral_envelope(sp, fs, options['mcep_dim'])
        # Aperiodicity; band-ap, 16KHz: 1, 48KHz: 5.
        ap = pw.d4c(x, f0, t, fs, fft_size=options['fft_size'])
        bap = pw.code_aperiodicity(ap, fs)

        # Save to the object.
        self.vocoder = 'WORLD'
        self.mcep = mcep
        self.f0 = f0
        self.bap = bap
        self.temporal_position = t
        self.vocoder_shift = options['shift']
        self.fft_size = options['fft_size']
        self.f0_floor = options['f0_floor']
        self.f0_ceil = options['f0_ceil']
        timestamp = datetime.datetime.fromtimestamp(time.time())
        self.timestamp = '%04d%02d%02d-%02d%02d%02d' % (
            timestamp.year, timestamp.month, timestamp.day, timestamp.hour,
            timestamp.minute, timestamp.second)
        self.pitch_tracker = options['pitch_tracker']

    def resynthesize(self) -> ndarray:
        """Re-synthesize the audio from f0, mcep, and bap.

        This re-synthesis process does not update the "wav" field.

        Returns:
            The re-synthesized waveform ranging from -1 to 1.
        """
        sp = pw.decode_spectral_envelope(self.mcep, self.fs, self.fft_size)
        ap = pw.decode_aperiodicity(self.bap, self.fs, self.fft_size)
        wav = pw.synthesize(self.f0, sp, ap, self.fs, self.vocoder_shift)
        return wav

    def write_audio(self, path):
        """Save the audio to the given path.

        Args:
            path: A path to a '.wav' file.
        """
        if self.wav.max() <= 1:  # If it is saved in float.
            wavfile.write(path, self.fs, self.wav)
        else:  # Saved in int16.
            wavfile.write(path, self.fs, self.wav.astype(np.int16))

    # The section below is reserved for getters and setters.
    @property
    def data(self) -> DataUtterance:
        return self._data

    @data.setter
    def data(self, val: DataUtterance):
        self._data.CopyFrom(val)

    @property
    def wav(self) -> ndarray:
        return mat_to_numpy(self._data.wav)

    @wav.setter
    def wav(self, val: ndarray):
        numpy_to_mat(val, self._data.wav)

    @property
    def fs(self) -> int:
        return self._data.fs

    @fs.setter
    def fs(self, val: int):
        # -1 is for the default case.
        if val > 0 or val == -1:
            self._data.fs = val
        else:
            raise ValueError("Sampling frequency must be positive!")

    @property
    def text(self) -> str:
        return self._data.text

    @text.setter
    def text(self, val: str):
        self._data.text = val

    @property
    def align(self) -> TextGrid:
        return read_tg_from_str(self._data.align)

    @align.setter
    def align(self, val: TextGrid):
        self._data.align = write_tg_to_str(val)

    @property
    def ppg(self) -> ndarray:
        return mat_to_numpy(self._data.ppg)

    @ppg.setter
    def ppg(self, val: ndarray):
        numpy_to_mat(val, self._data.ppg)

    @property
    def monophone_ppg(self) -> ndarray:
        return mat_to_numpy(self._data.monophone_ppg)

    @monophone_ppg.setter
    def monophone_ppg(self, val: ndarray):
        numpy_to_mat(val, self._data.monophone_ppg)

    @property
    def phone(self) -> IntervalTier:
        return read_segment(self._data.phone)

    @phone.setter
    def phone(self, val: IntervalTier):
        write_segment(val, self._data.phone)

    @property
    def word(self) -> IntervalTier:
        return read_segment(self._data.word)

    @word.setter
    def word(self, val: IntervalTier):
        write_segment(val, self._data.word)

    @property
    def lab(self) -> ndarray:
        return mat_to_numpy(self._data.lab)

    @lab.setter
    def lab(self, val: ndarray):
        val.astype(int)  # lab is saved in Int32Matrix format.
        numpy_to_mat(val, self._data.lab)

    @property
    def utterance_id(self) -> str:
        return self._data.utterance_id

    @utterance_id.setter
    def utterance_id(self, val: str):
        self._data.utterance_id = val

    @property
    def speaker_id(self) -> str:
        return self._data.meta_data.speaker_id

    @speaker_id.setter
    def speaker_id(self, val: str):
        self._data.meta_data.speaker_id = val

    @property
    def dialect(self) -> str:
        return MetaData.Dialect.Name(self._data.meta_data.dialect)

    @dialect.setter
    def dialect(self, val: str):
        """Set the dialect.

        Args:
            val: Must be one defined in MetaData.Dialect
        """
        self._data.meta_data.dialect = MetaData.Dialect.Value(val)

    @property
    def gender(self) -> str:
        return MetaData.Gender.Name(self._data.meta_data.gender)

    @gender.setter
    def gender(self, val: str):
        """Set the gender.

        Args:
            val: Must be one defined in MetaData.Gender
        """
        self._data.meta_data.gender = MetaData.Gender.Value(val)

    @property
    def original_file(self) -> str:
        return self._data.meta_data.original_file

    @original_file.setter
    def original_file(self, val: str):
        self._data.meta_data.original_file = val

    @property
    def num_channel(self) -> int:
        return self._data.meta_data.num_channel

    @num_channel.setter
    def num_channel(self, val: int):
        self._data.meta_data.num_channel = val

    @property
    def kaldi_shift(self) -> float:
        return self._data.kaldi_param.shift

    @kaldi_shift.setter
    def kaldi_shift(self, val: float):
        self._data.kaldi_param.shift = val

    @property
    def kaldi_window_size(self) -> float:
        return self._data.kaldi_param.window_size

    @kaldi_window_size.setter
    def kaldi_window_size(self, val: float):
        self._data.kaldi_param.window_size = val

    @property
    def kaldi_window_type(self) -> str:
        return self._data.kaldi_param.window_type

    @kaldi_window_type.setter
    def kaldi_window_type(self, val: str):
        self._data.kaldi_param.window_type = val

    @property
    def vocoder(self) -> str:
        return VocoderFeature.VocoderName.Name(self._data.vocoder_feat.vocoder)

    @vocoder.setter
    def vocoder(self, val: str):
        """Set the vocoder name.

        Args:
            val: Must be one defined in VocoderFeature.VocoderName
        """
        self._data.vocoder_feat.vocoder = VocoderFeature.VocoderName.Value(val)

    @property
    def spec(self) -> ndarray:
        return mat_to_numpy(self._data.vocoder_feat.filter.spec)

    @spec.setter
    def spec(self, val: ndarray):
        numpy_to_mat(val, self._data.vocoder_feat.filter.spec)
        self.spec_dim = self.spec.shape[1]
        self.fft_size = 2 * (self.spec_dim - 1)

    @property
    def mfcc(self) -> ndarray:
        return mat_to_numpy(self._data.vocoder_feat.filter.mfcc)

    @mfcc.setter
    def mfcc(self, val: ndarray):
        numpy_to_mat(val, self._data.vocoder_feat.filter.mfcc)
        self.mfcc_dim = self.mfcc.shape[1]

    @property
    def mcep(self) -> ndarray:
        return mat_to_numpy(self._data.vocoder_feat.filter.mcep)

    @mcep.setter
    def mcep(self, val: ndarray):
        numpy_to_mat(val, self._data.vocoder_feat.filter.mcep)
        self.mcep_dim = self.mcep.shape[1]

    @property
    def f0(self) -> ndarray:
        return mat_to_numpy(self._data.vocoder_feat.source.f0)

    @f0.setter
    def f0(self, val: ndarray):
        numpy_to_mat(val, self._data.vocoder_feat.source.f0)
        self.num_frame = self.f0.shape[0]

    @property
    def ap(self) -> ndarray:
        return mat_to_numpy(self._data.vocoder_feat.source.ap)

    @ap.setter
    def ap(self, val: ndarray):
        numpy_to_mat(val, self._data.vocoder_feat.source.ap)
        self.ap_dim = self.ap.shape[1]

    @property
    def bap(self) -> ndarray:
        return mat_to_numpy(self._data.vocoder_feat.source.bap)

    @bap.setter
    def bap(self, val: ndarray):
        numpy_to_mat(val, self._data.vocoder_feat.source.bap)
        if self.bap.ndim >= 2:
            self.bap_dim = self.bap.shape[1]
        else:
            self.bap_dim = 1

    @property
    def vuv(self) -> ndarray:
        return mat_to_numpy(self._data.vocoder_feat.source.vuv)

    @vuv.setter
    def vuv(self, val: ndarray):
        numpy_to_mat(val, self._data.vocoder_feat.source.vuv)

    @property
    def temporal_position(self) -> ndarray:
        return mat_to_numpy(self._data.vocoder_feat.source.temporal_position)

    @temporal_position.setter
    def temporal_position(self, val: ndarray):
        numpy_to_mat(val, self._data.vocoder_feat.source.temporal_position)

    @property
    def vocoder_window_size(self) -> float:
        return self._data.vocoder_feat.param.window_size

    @vocoder_window_size.setter
    def vocoder_window_size(self, val: float):
        self._data.vocoder_feat.param.window_size = val

    @property
    def vocoder_window_type(self) -> str:
        return self._data.vocoder_feat.param.window_type

    @vocoder_window_type.setter
    def vocoder_window_type(self, val: str):
        self._data.vocoder_feat.param.window_type = val

    @property
    def vocoder_shift(self) -> float:
        return self._data.vocoder_feat.param.shift

    @vocoder_shift.setter
    def vocoder_shift(self, val: float):
        self._data.vocoder_feat.param.shift = val

    @property
    def num_frame(self) -> int:
        return self._data.vocoder_feat.param.num_frame

    @num_frame.setter
    def num_frame(self, val: int):
        self._data.vocoder_feat.param.num_frame = val

    @property
    def alpha(self) -> float:
        return self._data.vocoder_feat.param.alpha

    @alpha.setter
    def alpha(self, val: float):
        self._data.vocoder_feat.param.alpha = val

    @property
    def fft_size(self) -> int:
        return self._data.vocoder_feat.param.fft_size

    @fft_size.setter
    def fft_size(self, val: int):
        self._data.vocoder_feat.param.fft_size = val

    @property
    def spec_dim(self) -> int:
        return self._data.vocoder_feat.param.spec_dim

    @spec_dim.setter
    def spec_dim(self, val: int):
        self._data.vocoder_feat.param.spec_dim = val

    @property
    def mfcc_dim(self) -> int:
        return self._data.vocoder_feat.param.mfcc_dim

    @mfcc_dim.setter
    def mfcc_dim(self, val: int):
        self._data.vocoder_feat.param.mfcc_dim = val

    @property
    def mcep_dim(self) -> int:
        return self._data.vocoder_feat.param.mcep_dim

    @mcep_dim.setter
    def mcep_dim(self, val: int):
        self._data.vocoder_feat.param.mcep_dim = val

    @property
    def f0_floor(self) -> float:
        return self._data.vocoder_feat.param.f0_floor

    @f0_floor.setter
    def f0_floor(self, val: float):
        self._data.vocoder_feat.param.f0_floor = val

    @property
    def f0_ceil(self) -> float:
        return self._data.vocoder_feat.param.f0_ceil

    @f0_ceil.setter
    def f0_ceil(self, val: float):
        self._data.vocoder_feat.param.f0_ceil = val

    @property
    def timestamp(self) -> str:
        return self._data.vocoder_feat.param.timestamp

    @timestamp.setter
    def timestamp(self, val: str):
        self._data.vocoder_feat.param.timestamp = val

    @property
    def ap_dim(self) -> int:
        return self._data.vocoder_feat.param.ap_dim

    @ap_dim.setter
    def ap_dim(self, val: int):
        self._data.vocoder_feat.param.ap_dim = val

    @property
    def bap_dim(self) -> int:
        return self._data.vocoder_feat.param.bap_dim

    @bap_dim.setter
    def bap_dim(self, val: int):
        self._data.vocoder_feat.param.bap_dim = val

    @property
    def pitch_tracker(self) -> str:
        return self._data.vocoder_feat.param.pitch_tracker

    @pitch_tracker.setter
    def pitch_tracker(self, val: str):
        self._data.vocoder_feat.param.pitch_tracker = val
