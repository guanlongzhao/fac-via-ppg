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

from common.hparams import create_hparams_stage
from script.train import load_model
from common.utils import to_gpu
from common.utils import notch_filtering
from common.layers import TacotronSTFT
from common import feat
from scipy.io import wavfile
import sys
import numpy as np
import torch
import ppg
import os
import logging
import datetime
import time
sys.path.append('/home/guanlong/PycharmProjects/ppg-speech/src/waveglow')
from waveglow.denoiser import Denoiser


def get_mel(wav, stft):
    audio = torch.FloatTensor(wav.astype(np.float32))
    audio_norm = audio / 32768
    audio_norm = audio_norm.unsqueeze(0)
    audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    # (1, n_mel_channels, T)
    acoustic_feats = stft.mel_spectrogram(audio_norm)
    return acoustic_feats


def waveglow_audio(mel, waveglow, sigma, is_cuda_output=False):
    mel = torch.autograd.Variable(mel.cuda())
    if not is_cuda_output:
        with torch.no_grad():
            audio = 32768 * waveglow.infer(mel, sigma=sigma)[0]
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
    else:
        with torch.no_grad():
            audio = waveglow.infer(mel, sigma=sigma).cuda()
    return audio


def get_inference(seq, model, is_clip=False):
    """Tacotron inference.

    Args:
        seq: T*D numpy array.
        model: Tacotron model.
        is_clip: Set to True to avoid the artifacts at the end.

    Returns:
        synthesized mels.
    """
    # (T, D) numpy -> (1, D, T) cpu tensor
    seq = torch.from_numpy(seq).float().transpose(0, 1).unsqueeze(0)
    # cpu tensor -> gpu tensor
    seq = to_gpu(seq)
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(seq)
    if is_clip:
        return mel_outputs_postnet[:, :, 10:(seq.size(2)-10)]
    else:
        return mel_outputs_postnet


def load_waveglow_model(path):
    model = torch.load(path)['model']
    model = model.remove_weightnorm(model)
    model.cuda().eval()
    return model


def get_ppg(wav_path, deps, is_fmllr=False, speaker_code=None):
    fs, wav = wavfile.read(wav_path)
    wave_data = feat.read_wav_kaldi_internal(wav, fs)
    if is_fmllr:
        seq = ppg.compute_full_ppg_wrapper(wave_data, deps.nnet, deps.lda, 10,
                                           fmllr=deps.fmllr_mats[speaker_code])
    else:
        seq = ppg.compute_full_ppg_wrapper(wave_data, deps.nnet, deps.lda, 10)
    return seq


if __name__ == '__main__':
    # Prepare dirs
    timestamp = datetime.datetime.fromtimestamp(time.time())
    output_dir = \
        '/media/guanlong/DATA1/exp/ppg-speech/samples/trial_%04d%02d%02d' \
        '-%02d%02d%02d' \
        % (timestamp.year, timestamp.month, timestamp.day, timestamp.hour,
           timestamp.minute, timestamp.second)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, 'ac'))
        os.mkdir(os.path.join(output_dir, 'inference'))
    logging.basicConfig(filename=os.path.join(output_dir, 'debug.log'),
                        level=logging.DEBUG)
    logging.info('Output dir: %s', output_dir)

    # Parameters
    exp_info = ''
    checkpoint_path = ''
    # Use None for SI AM.
    am_path = None
    fmllr_speaker_name = None  # 'spk' | None (no fmllr, SI model)
    teacher_utts_root = '/media/guanlong/DATA1/GDriveTAMU/PSI-Lab/corpora/' \
                        'arctic/XXX/recordings'
    student_utts_root = '/media/guanlong/DATA1/GDriveTAMU/PSI-Lab/corpora/' \
                        'gsb16k/YYY/recordings_noise_reduced'
    teacher_file_prefix = 'arctic_'  # Change to '' if no prefix
    student_file_prefix = 'gsb_'  # Change to '' if no prefix

    waveglow_path = ''
    is_clip = False  # Set to True to control the output length of AC.
    fs = 16000
    waveglow_sigma = 0.6
    test_utt_indices = range(1083, 1133)
    num_utts = len(test_utt_indices)
    # Parameters for notch filtering.
    # Set to None to disable. [2000, 4000, 6000] for 16kHz
    filter_freqs = None
    Q = 50
    # Parameters for official waveglow denoising.
    is_official_denoiser = True
    if is_official_denoiser:
        waveglow_for_denoiser = torch.load(waveglow_path)['model']
        waveglow_for_denoiser.cuda()
        denoiser_mode = 'zeros'
        denoiser = Denoiser(waveglow_for_denoiser, mode=denoiser_mode)
        denoiser_strength = 0.005
    else:
        denoiser = None
        denoiser_strength = 0
        denoiser_mode = ''
    is_self_inference = False
    # End of parameters

    logging.info('Info: %s', exp_info)
    logging.debug('Tacotron: %s', checkpoint_path)
    logging.debug('Waveglow: %s', waveglow_path)
    if am_path:
        logging.debug('AM: %s', am_path)
    else:
        logging.debug('AM: SI model')
    logging.debug('is_clip: %d', is_clip)
    logging.debug('Fs: %d', fs)
    logging.debug('Sigma: %f', waveglow_sigma)
    if fmllr_speaker_name:
        logging.debug('fMLLR name: %s', fmllr_speaker_name)
    else:
        logging.debug('fMLLR name: None')
    logging.info('Number of test utts: %d', num_utts)
    logging.info('Teacher root dir: %s', teacher_utts_root)
    logging.info('Student root dir: %s', student_utts_root)
    if filter_freqs and not is_official_denoiser:
        logging.debug('Notch filtering freqs: [%s]',
                      ', '.join([str(freq) for freq in filter_freqs]))
        logging.debug('Notch filtering Q: %f', Q)
    else:
        logging.debug('Notch filtering freqs: None')
    logging.debug('is_official_denoiser: %d', is_official_denoiser)
    logging.debug('Denoiser strength: %f', denoiser_strength)
    logging.debug('Denoiser mode: %s', denoiser_mode)
    logging.debug('is_self_inference: %d', is_self_inference)

    hparams = create_hparams_stage()
    taco_stft = TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_acoustic_feat_dims, hparams.sampling_rate,
        hparams.mel_fmin, hparams.mel_fmax)

    # Load models.
    tacotron_model = load_model(hparams)
    tacotron_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = tacotron_model.eval()
    waveglow_model = load_waveglow_model(waveglow_path)

    if am_path:
        deps = ppg.DependenciesPPG(nnet_path=am_path)
        is_fmllr = True
    else:
        deps = ppg.DependenciesPPG()
        is_fmllr = False
    logging.debug('is_fmllr: %d', is_fmllr)

    processed_utts = 0
    for ii in test_utt_indices:
        processed_utts += 1
        logging.info('Processing %02d/%02d', processed_utts, num_utts)
        teacher_utt_path = os.path.join(teacher_utts_root,
                                        '%s%04d.wav' % (teacher_file_prefix,
                                                        ii))
        student_utt_path = os.path.join(student_utts_root, '%s%04d.wav' % (
            student_file_prefix, ii))

        if os.path.isfile(teacher_utt_path):
            logging.info('Perform AC on %s', teacher_utt_path)
            teacher_ppg = get_ppg(teacher_utt_path, deps, is_fmllr,
                                  speaker_code=fmllr_speaker_name)
            ac_mel = get_inference(teacher_ppg, tacotron_model, is_clip)

            if is_official_denoiser:
                ac_wav = waveglow_audio(ac_mel, waveglow_model,
                                        waveglow_sigma, True)
                ac_wav = denoiser(
                    ac_wav, strength=denoiser_strength)[:, 0].cpu().numpy().T
            else:
                ac_wav = waveglow_audio(ac_mel, waveglow_model, waveglow_sigma)

            output_file = os.path.join(output_dir, 'ac', '%04d.wav' % ii)
            wavfile.write(output_file, fs, ac_wav)

            if filter_freqs and not is_official_denoiser:
                fs, ac_wav = wavfile.read(output_file)
                for w0 in filter_freqs:
                    ac_wav = notch_filtering(ac_wav, fs, w0, Q)
                wavfile.write(output_file, fs, ac_wav.astype(np.int16))
        else:
            logging.warning('Missing %s', teacher_utt_path)
        if is_self_inference:
            if os.path.isfile(student_utt_path):
                logging.info('Perform self-inference on %s', student_utt_path)
                student_ppg = get_ppg(student_utt_path, deps, is_fmllr,
                                      speaker_code=fmllr_speaker_name)
                infer_mel = get_inference(student_ppg, tacotron_model)

                if is_official_denoiser:
                    infer_wav = waveglow_audio(infer_mel, waveglow_model,
                                               waveglow_sigma, True)
                    infer_wav = denoiser(
                        infer_wav,
                        strength=denoiser_strength)[:, 0].cpu().numpy().T
                else:
                    infer_wav = waveglow_audio(infer_mel, waveglow_model,
                                               waveglow_sigma)

                output_file = os.path.join(output_dir,
                                           'inference', '%04d.wav' % ii)
                wavfile.write(output_file, fs, infer_wav)

                if filter_freqs and not is_official_denoiser:
                    fs, infer_wav = wavfile.read(output_file)
                    for w0 in filter_freqs:
                        infer_wav = notch_filtering(infer_wav, fs, w0, Q)
                    wavfile.write(output_file, fs, infer_wav.astype(np.int16))
            else:
                logging.warning('Missing %s', student_utt_path)

    logging.info('Done!')
