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

from io import StringIO
import logging
import os
import re
import subprocess
import tempfile
from glob import glob
from numpy import ndarray
from scipy.io import wavfile
from shutil import move
from textgrid import TextGrid, PointTier, IntervalTier, Interval, Point

DEFAULT_TEXTGRID_PRECISION = 5


class MontrealAligner(object):
    """Class to perform forced alignment using the montreal aligner.

    See https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner for
    details about the montreal aligner. This class is based on the v1.0.0
    release and therefore might not work with other versions.
    You will need to configure the default settings of the aligner, i.e., the
    location of the tool, the acoustic model, and lexicon. You might also need
    to compile the montreal aligner for your platform since the official site
    only provides the compiled version for macOS, Windows, and Ubuntu 16.04.

    Sample usage,
        aligner = MontrealAligner()
        # Single utterance
        tg = aligner.align_single("test.wav", "test.lab")
        # Batch processing
        output_dir = aligner.align_batch("input/path", "output/path")
    """
    def __init__(self, aligner_path="/var/montreal-forced-aligner-1.0.0/bin/"
                                    "mfa_align",
                 am_path="/var/montreal-forced-aligner-1.0.0/pretrained_models/"
                         "english.zip",
                 dict_path="/var/montreal-forced-aligner-1.0.0/"
                           "pretrained_models/dictionary"):
        """Create an aligner instance.

        Args:
            aligner_path: The path to the aligner binary file (mfa_align).
            am_path: The path to the acoustic model (language.zip).
            dict_path: The path to the lexicon file (dictionary).
        """
        if not os.path.isfile(aligner_path):
            logging.error("Aligner %s does not exist!", aligner_path)
        self.aligner_path = aligner_path
        if not os.path.isfile(am_path):
            logging.error("Model %s does not exist!", am_path)
        self.am_path = am_path
        if not os.path.isfile(dict_path):
            logging.error("Lexicon %s does not exist!", dict_path)
        self.dict_path = dict_path
        self.filter_transcript = re.compile(r"[^0-9a-zA-Z,.\s'-]")

    def align_single_internal(self, wav: ndarray,
                              fs: int, text: str) -> TextGrid:
        """Internal wrapper for aligning a single utterance.

        This method does not perform speaker adaptation since it does not make
        sense to adapt on only one arbitrary sentence. The method will also
        clean up the input text transcription before the alignment. All
        intermediate files will be stored in a temp folder created on-the-fly in
        the system "tmp" folder, and will be cleaned after the alignment
        finishes.

        Args:
            wav: Speech data.
            fs: Sampling frequency of the wave data.
            text: The text transcription of the speech.

        Returns:
            text_grid: A TextGrid object.
        """
        # Save all intermediate files in a temp folder.
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_wav_path = os.path.join(tmpdirname, "test.wav")
            wav = wav.astype('int16')
            wavfile.write(temp_wav_path, fs, wav)
            temp_txt_path = os.path.join(tmpdirname, "test.lab")
            # Get rid of invalid characters.
            text = self.filter_transcript.sub("", text)
            with open(temp_txt_path, 'w') as transcript:
                transcript.write(text)
            temp_align_dir = os.path.join(tmpdirname, "temp_align")
            os.mkdir(temp_align_dir)
            temp_output_dir = os.path.join(tmpdirname, "temp_output")
            # Run in single utterance mode, no need to do adaptation (-n).
            cmd = [self.aligner_path, "-t", temp_align_dir, "-j", "1", "-v",
                   "-n", "-c", "-i", tmpdirname, self.dict_path, self.am_path,
                   temp_output_dir]
            # if the process exits with a non-zero exit code, a
            # "CalledProcessError" exception will be raised.
            run_status = subprocess.run(cmd, check=True)
            temp_basename = os.path.basename(tmpdirname)
            temp_tg_path = os.path.join(temp_output_dir, temp_basename,
                                        "test.TextGrid")
            text_grid = TextGrid()
            text_grid.read(temp_tg_path)
        return text_grid

    def align_single(self, wav, text, fs=None, is_wav_path=True,
                     is_text_path=True) -> TextGrid:
        """ This is the I/O wrapper for "align_single_internal."

        Sample usage,
            # Default
            tg = aligner.align_single("test.wav", "test.lab")
            # Input a wave file and its transcription in a string
            tg = aligner.align_single("test.wav", "test", is_text_path=False)
            # Directly input numpy format wave data whose content is the word
            # "test" and has a sampling frequency of 16KHz; the flags tell the
            # method that the inputs are not file paths
            tg = aligner.align_single(wave_data, "test", 16000, False, False)

        Args:
            wav: Path to a wav file or a numpy array containing the speech data.
            text: Path to a text file or a string containing the transcript.
            fs: [optional] Sampling frequency. Only set if "wav" is a np array.
            is_wav_path: [optional] Set to False if "wav" is a np array.
            is_text_path: [optional] Set to False if "text" is a normal string.

        Returns:
            text_grid: A TextGrid object.
        """
        # Prepare speech data.
        if is_wav_path:
            if not os.path.isfile(wav):
                logging.error("Wave file %s does not exist!", wav)
            fs, wav_val = wavfile.read(wav, False)
        else:
            if fs is None:
                logging.error("Need to specify sampling frequency!")
            wav_val = wav
        assert fs is not None, "Sampling frequency not determined."
        assert isinstance(wav_val, ndarray), "Check your input for wav."
        # Prepare transcription data.
        if is_text_path:
            if not os.path.isfile(text):
                logging.error("Text file %s does not exist!", text)
            with open(text, 'r') as reader:
                text_val = reader.readline()
            if not text_val:
                logging.error("Empty text file %s!", text)
        else:
            text_val = text
        assert isinstance(text_val, str), "Check your input for text."
        # Run alignment.
        text_grid = self.align_single_internal(wav_val, fs, text_val)
        return text_grid

    def align_batch(self, input_dir, output_dir, num_jobs=4):
        """Wrapper for the normal montreal aligner usage.

        Processing a group of utterances in "input_dir," the files should be
        organized following the normal montreal aligner convention, i.e., each
        wave file will have an accompanying .lab file with the same name that
        contains the text transcription.

        Args:
            input_dir: Path to the audio and transcription files.
            output_dir: Path to the output dir that will contain all the TGs.
            num_jobs: Number of parallel threads.

        Returns:
            output_dir: Path to the output dir.
        """
        if not os.path.isdir(input_dir):
            logging.error("Input dir %s does not exist!", input_dir)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        with tempfile.TemporaryDirectory() as tmpdirname:
            cmd = [self.aligner_path, "-t", tmpdirname, "-j", str(num_jobs),
                   "-v", "-c", input_dir, self.dict_path, self.am_path,
                   output_dir]
            # if the process exits with a non-zero exit code, a
            # "CalledProcessError" exception will be raised.
            run_status = subprocess.run(cmd, check=True)
        # Move the files around, since the aligner will put the tg files in a
        # sub-folder in "output_dir".
        output_basename = os.path.basename(input_dir)
        actual_output_dir = os.path.join(output_dir, output_basename)
        for each_file in glob(os.path.join(actual_output_dir, "*.TextGrid")):
            move(each_file, output_dir)
        return output_dir


def write_tg_to_str(tg, null=''):
    """Write the current TextGrid into a Praat-format TextGrid string.

    Adapted from TextGrid.write()
    """
    if not isinstance(tg, TextGrid):
        logging.warning("Alignment does not exist!")
        return None
    sink = StringIO()
    print('File type = "ooTextFile"', file=sink)
    print('Object class = "TextGrid"\n', file=sink)
    print('xmin = {0}'.format(tg.minTime), file=sink)
    # Compute max time
    max_t = tg.maxTime
    if not max_t:
        max_t = max([t.maxTime if t.maxTime else t[-1].maxTime
                     for t in tg.tiers])
    print('xmax = {0}'.format(max_t), file=sink)
    print('tiers? <exists>', file=sink)
    print('size = {0}'.format(len(tg)), file=sink)
    print('item []:', file=sink)
    for (i, tier) in enumerate(tg.tiers, 1):
        print('\titem [{0}]:'.format(i), file=sink)
        if tier.__class__ == IntervalTier:
            print('\t\tclass = "IntervalTier"', file=sink)
            print('\t\tname = "{0}"'.format(tier.name), file=sink)
            print('\t\txmin = {0}'.format(tier.minTime), file=sink)
            print('\t\txmax = {0}'.format(max_t), file=sink)
            # Compute the number of intervals and make the empty ones
            output = tier._fillInTheGaps(null)
            print('\t\tintervals: size = {0}'.format(
                len(output)), file=sink)
            for (j, interval) in enumerate(output, 1):
                print('\t\t\tintervals [{0}]:'.format(j), file=sink)
                print('\t\t\t\txmin = {0}'.format(
                    interval.minTime), file=sink)
                print('\t\t\t\txmax = {0}'.format(
                    interval.maxTime), file=sink)
                mark = interval.mark.replace('"', '""')
                print('\t\t\t\ttext = "{0}"'.format(mark), file=sink)
        elif tier.__class__ == PointTier:  # PointTier
            print('\t\tclass = "TextTier"', file=sink)
            print('\t\tname = "{0}"'.format(tier.name), file=sink)
            print('\t\txmin = {0}'.format(tier.minTime), file=sink)
            print('\t\txmax = {0}'.format(max_t), file=sink)
            print('\t\tpoints: size = {0}'.format(len(tier)), file=sink)
            for (k, point) in enumerate(tier, 1):
                print('\t\t\tpoints [{0}]:'.format(k), file=sink)
                print('\t\t\t\ttime = {0}'.format(point.time), file=sink)
                mark = point.mark.replace('"', '""')
                print('\t\t\t\tmark = "{0}"'.format(mark), file=sink)
    tg_text = sink.getvalue()
    sink.close()
    return tg_text


def parse_line(line, short, to_round):
    """Copied from textgrid.parse_line"""
    line = line.strip()
    if short:
        if '"' in line:
            return line[1:-1]
        return round(float(line), to_round)
    if '"' in line:
        m = re.match(r'.+? = "(.*)"', line)
        return m.groups()[0]
    m = re.match(r'.+? = (.*)', line)
    return round(float(m.groups()[0]), to_round)


def parse_header(source):
    """Copied from textgrid.parse_header"""
    header = source.readline()  # header junk
    m = re.match('File type = "([\w ]+)"', header)
    if m is None or not m.groups()[0].startswith('ooTextFile'):
        raise ValueError('The file could not be parsed as a Praat text file as '
                         'it is lacking a proper header.')

    short = 'short' in m.groups()[0]
    file_type = parse_line(source.readline(), short, '')  # header junk
    t = source.readline()  # header junk
    return file_type, short


def get_mark(text, short):
    """
    Return the mark or text entry on a line. Praat escapes double-quotes
    by doubling them, so doubled double-quotes are read as single
    double-quotes. Newlines within an entry are allowed.

    Copied from textgrid._getMark
    """

    line = text.readline()

    # check that the line begins with a valid entry type
    if not short and not re.match(r'^\s*(text|mark) = "', line):
        raise ValueError('Bad entry: ' + line)

    # read until the number of double-quotes is even
    while line.count('"') % 2:
        next_line = text.readline()

        if not next_line:
            raise EOFError('Bad entry: ' + line[:20] + '...')

        line += next_line
    if short:
        pattern = r'^"(.*?)"\s*$'
    else:
        pattern = r'^\s*(text|mark) = "(.*?)"\s*$'
    entry = re.match(pattern, line, re.DOTALL)

    return entry.groups()[-1].replace('""', '"')


def read_tg_from_str(tg_str, round_digits=DEFAULT_TEXTGRID_PRECISION):
    """
    Read the tiers contained in the Praat-formatted string tg_str into a
    TextGrid object.
    Times are rounded to the specified precision.

    Adapted from TextGrid.read()
    """
    source = StringIO(tg_str)
    tg = TextGrid()

    file_type, short = parse_header(source)
    if file_type != "TextGrid":
        raise ValueError("The file could not be parsed as a TextGrid as it is "
                         "lacking a proper header.")
    tg.minTime = parse_line(source.readline(), short, round_digits)
    tg.maxTime = parse_line(source.readline(), short, round_digits)
    source.readline()  # More header junk
    if short:
        m = int(source.readline().strip())  # Will be tg.n
    else:
        m = int(source.readline().strip().split()[2])  # Will be tg.n
    if not short:
        source.readline()
    for i in range(m):  # Loop over grids
        if not short:
            source.readline()
        if parse_line(source.readline(), short,
                      round_digits) == "IntervalTier":
            inam = parse_line(source.readline(), short, round_digits)
            imin = parse_line(source.readline(), short, round_digits)
            imax = parse_line(source.readline(), short, round_digits)
            itie = IntervalTier(inam, imin, imax)
            itie.strict = tg.strict
            n = int(parse_line(source.readline(), short, round_digits))
            for j in range(n):
                if not short:
                    source.readline().rstrip().split()  # Header junk
                jmin = parse_line(source.readline(), short, round_digits)
                jmax = parse_line(source.readline(), short, round_digits)
                jmrk = get_mark(source, short)
                if jmin < jmax:  # Non-null
                    itie.addInterval(Interval(jmin, jmax, jmrk))
            tg.append(itie)
        else:  # PointTier
            inam = parse_line(source.readline(), short, round_digits)
            imin = parse_line(source.readline(), short, round_digits)
            imax = parse_line(source.readline(), short, round_digits)
            itie = PointTier(inam)
            n = int(parse_line(source.readline(), short, round_digits))
            for j in range(n):
                source.readline().rstrip()  # Header junk
                jtim = parse_line(source.readline(), short, round_digits)
                jmrk = get_mark(source, short)
                itie.addPoint(Point(jtim, jmrk))
            tg.append(itie)
    return tg
