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
import os


def prep_wav(orig_wav, out_wav, sr, new_sr):
    print("Resampling wav file from " + str(sr) + " to " + str(new_sr) + "...")
    os.system("sox " + orig_wav + " -r " + str(new_sr) + " " + out_wav)
    return True


if __name__ == '__main__':
    in_dir = ''
    out_dir = ''
    in_fs = 44100
    out_fs = 16000
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for in_file in os.listdir(in_dir):
        if in_file.endswith('.wav'):
            prep_wav(os.path.join(in_dir, in_file),
                     os.path.join(out_dir, in_file), in_fs, out_fs)
