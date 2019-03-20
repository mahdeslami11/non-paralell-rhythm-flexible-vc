import sys
import os
import yaml
import tgt
import librosa
import pickle

from ..src.utils import AudioProcessor

config = yaml.load(open('../config/config.yaml', 'r'))

ap_kwargs = config["audio"]
ap = AudioProcessor(**ap_kwargs)

align_res_path = config['path']['align_result']
data_dir = config['path']['all_data_dir']
feat_dir = config['path']['feat_dir']
meta_path = config['path']['meta_path']
meta = open(meta_path, 'w')

for dirPath, dirNames, fileNames in os.walk(align_res_path):
    for f in fileNames:
        f_id = f.replace('.Textgrid', '')
        tgtpath = os.path.join(dirPath, f)
        tg = tgt.io.read_textgrid(tgtpath)
        wavpath = os.path.join(data_dir, tgtname.strip().split('/')[-1]).replace('.Textgrid', '.wav')
        textpath = os.path.join(data_dir, tgtname.strip().split('/')[-1]).replace('.Textgrid', '.txt')
        featpath = os.path.join(feat_dir, )

        wav = ap.load_wav(wavpath)
        mag = ap.spectrogram(wav)
        mel = ap.melspectrogram(wav)
        MC, f0, AP = ap.get_MCEPs(wav)
        print(mag.shape, mel.shape, MC.shape, f0.shape, AP.shape)
        text = open(textpath).readline().strip().lower()
        print(text)

        # sentence level (one sentence to many phns)
        if s_level:
            phn_bounds = tg.get_tier_by_name('phones').intervals


            for phn_bound in phn_bounds:
                s_t = int(float(phn_bound.start_time) * sample_rate)
                e_t = int(float(phn_bound.end_time) * sample_rate)
                print(s_t, e_t, phn_bound.text)

            meta.write("{}|{}|{}\n".format(f_id, text, featpath,))
        exit()

'''
# word level (one word to many phns)
if w_level:
    word_bounds = tg.get_tier_by_name('words').intervals

    for idx, word_bound in enumerate(word_bounds):
        phns = tg.get_tier_by_name('phones')
        phn_bounds = phns.get_annotations_between_timepoints(
            word_bound.start_time,
            word_bound.end_time
        )
        for phn_bound in phn_bounds:
            p_s_t = int(float(phn_bound.start_time) * sample_rate) - w_s_t
            p_e_t = int(float(phn_bound.end_time) * sample_rate) - w_s_t
            print(p_s_t, p_e_t, phn_bound.text)
'''
'''
ap = AudioProcessor(
        sample_rate=a_config["sample_rate"],
        n_mels=a_config["n_mels"],
        n_fft=a_config["n_fft"],
        frame_length_ms=a_config["frame_length_ms"],
        frame_shift_ms=a_config["frame_shift_ms"],
        preemphasis=a_config["preemphasis"],
        min_level_db=a_config["min_level_db"],
        ref_level_db=a_config["ref_level_db"],
        griffin_lim_iters=a_config["griffin_lim_iters"],
        power=a_config["power"],
        order=a_config["order"],
        alpha=a_config["alpha"]
    )
'''
