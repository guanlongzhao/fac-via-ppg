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
import re
from textgrid import TextGrid, PointTier, IntervalTier, Interval, Point

DEFAULT_TEXTGRID_PRECISION = 5


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
