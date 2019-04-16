import sys
import yaml
from src.utils import AudioProcessor

ap=AudioProcessor(**yaml.load(open('./config/config.yaml'))['audio'])

wav = ap.load_wav(sys.argv[1])
ap.save_wav(wav, './gt.wav')

mag, _ = ap.get_spec(wav)
wav_hat = ap.inv_spectrogram(mag)

ap.save_wav(wav_hat, sys.argv[2])
