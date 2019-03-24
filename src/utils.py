import os
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import librosa
import librosa.filters
import pyworld as pw
import pysptk
from scipy import signal

class AudioProcessor(object):
    def __init__(self,
        sample_rate, n_mels, n_fft, frame_length_ms, frame_shift_ms,
        preemphasis, min_level_db, ref_level_db,
        griffin_lim_iters, power, order, alpha):

        self.sr = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = int(frame_shift_ms / 1000 * sample_rate)
        self.win_length = int(frame_length_ms / 1000 * sample_rate)
        self.preemph = preemphasis
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db
        self.GL_iter = griffin_lim_iters
        self.mel_basis = librosa.filters.mel(self.sr, self.n_fft, n_mels=self.n_mels)
        self.power = power
        self.order = order
        self.alpha = alpha

    def load_wav(self, path):
        return librosa.core.load(path, sr=self.sr)[0]

    def save_wav(self, wav, path):
        librosa.output.write_wav(path, wav, self.sr)
        return

    def preemphasis(self, wav):
        return signal.lfilter([1, -self.preemph], [1], wav)

    def inv_preemphasis(self, wav_preemph):
        return signal.lfilter([1], [1, -self.preemph], wav_preemph)

    def spectrogram(self, wav):
        D = self._stft(self.preemphasis(wav))
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db
        return self._normalize(S).T

    def inv_spectrogram(self, linear_spect):
        '''Converts spectrogram to waveform using librosa'''
        linear_spect = linear_spect.T
        S = self._db_to_amp(self._denormalize(linear_spect) + self.ref_level_db)
        return self.inv_preemphasis(self._griffin_lim(S ** self.power))

    def melspectrogram(self, wav):
        D = self._stft(self.preemphasis(wav))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D)))
        return self._normalize(S).T

    def get_spec(self, wav):
        ''' get mag and mel at the same time'''
        D = self._stft(self.preemphasis(wav))
        mag_sp = np.abs(D)
        mel_sp = self._linear_to_mel(mag_sp)

        mag = self._normalize(self._amp_to_db(mag_sp) - self.ref_level_db).T
        mel = self._normalize(self._amp_to_db(mel_sp)).T

        return mag, mel

    def _griffin_lim(self, S):
        '''librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        '''
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for i in range(self.GL_iter):
          angles = np.exp(1j * np.angle(self._stft(y)))
          y = self._istft(S_complex * angles)
        return y.astype(np.float32)

    def _stft(self, x):
        return librosa.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

    def _istft(self, x):
        return librosa.istft(x, hop_length=self.hop_length, win_length=self.win_length)

    def _linear_to_mel(self, linear_spect):
        return np.dot(self.mel_basis, linear_spect)

    def _amp_to_db(self, x):
        return 20 * np.log10(np.maximum(1e-5, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _normalize(self, x):
        return np.clip((x - self.min_level_db) / -self.min_level_db, 0, 1)

    def _denormalize(self, x):
        return (np.clip(x, 0, 1) * -self.min_level_db) + self.min_level_db

    def get_MCEPs(self, wav):
        wav = np.float64(wav)
        _f0_h, t_h = pw.harvest(wav, self.sr)
        f0_h = pw.stonemask(wav, _f0_h, t_h, self.sr)
        sp_h = pw.cheaptrick(wav, f0_h, t_h, self.sr)
        ap_h = pw.d4c(wav, f0_h, t_h, self.sr)
        mc = pysptk.sp2mc(sp_h, order=self.order, alpha=self.alpha)

        return mc, f0_h, ap_h

    def MCEPs2wav(self, mc, f0, ap):
        sp = pysptk.mc2sp(np.float64(mc),alpha=self.alpha, fftlen=self.n_fft)
        y = pw.synthesize(np.float64(f0), np.float64(sp), np.float64(ap), self.sr, pw.default_frame_period)

        return y.astype(np.float32)

def plot_alignment(alignment, gs, idx):
    fig, ax = plt.subplots()
    im = ax.imshow(alignment)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}_{}k.png'.format(hp.log_dir, idx, gs//1000), format='png')

def my_shuffle(*args):
    randomize = np.arange(len(args[0]))
    np.random.shuffle(randomize)
    res = [x[randomize] for x in args]
    if len(res) >= 2:
        return res
    else:
        return res[0]
