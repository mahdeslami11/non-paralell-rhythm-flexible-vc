import sys
sys.path.insert(0,'..')
import os
import yaml
import tgt
import librosa
import pickle
import re
import unicodedata

from src.utils import AudioProcessor

def text_normalize(text, char_set):
    # Strip accents
    text = ''.join(char for char in unicodedata.normalize('NFD', text) \
        if unicodedata.category(char) != 'Mn')
    text = text.lower()
    text = re.sub("[^{}]".format(char_set), " ", text)
    text = re.sub("[ ]+", " ", text).strip()
    return text

def get_per_frame_phn(ap, tg, mag_len):
    endoff_phn_list = []
    phn_bounds = tg.get_tier_by_name('phones').intervals
    for phn_bound in phn_bounds:
        s_t = int(float(phn_bound.start_time) * ap.sr)
        e_t = int(float(phn_bound.end_time) * ap.sr)
        print(s_t, e_t, phn_bound.text)
        endoff_phn_list.append((e_t, phn_bound.text))

    per_frame_phn = []
    end_idx = 0
    for m_idx in range(mag_len):
        offset = m_idx * ap.hop_length
        if offset < endoff_phn_list[end_idx][0]:
            per_frame_phn.append(endoff_phn_list[end_idx][1])
        else:
            end_idx = min(end_idx+1, len(endoff_phn_list)-1)
            per_frame_phn.append(endoff_phn_list[end_idx][1])
    assert len(per_frame_phn) == mag_len

    return per_frame_phn

def main():
    config = yaml.load(open('../config/config.yaml', 'r'))
    ap_kwargs = config["audio"]
    ap = AudioProcessor(**ap_kwargs)

    align_res_path = config['path']['align_result']
    data_dir = config['path']['all_data_dir']
    feat_dir = config['path']['feat_dir']
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    char_set = config['text']['char_set']
    meta_path = config['path']['meta_path']
    all_meta = []

    # for acoustic features
    for dirPath, dirNames, fileNames in os.walk(align_res_path):
        for f in fileNames:
            f_id = f.replace('.TextGrid', '')
            tgtpath = os.path.join(dirPath, f)
            wavpath = os.path.join(data_dir, f_id+'.wav')
            textpath = os.path.join(data_dir, f_id+'.lab')
            featpath = os.path.join(feat_dir, f_id+'.pkl')

            wav = ap.load_wav(wavpath)
            mag, mel = ap.get_spec(wav)
            MC, f0, AP = ap.get_MCEPs(wav)

            text = open(textpath).readline().strip().lower()
            text = text_normalize(text, char_set)
            tg = tgt.io.read_textgrid(tgtpath)
            per_frame_phn = get_per_frame_phn(ap, tg, len(mag))

            feat = {
                'f_id': f_id,
                'mag': mag, 'mel': mel,
                'MC': MC, 'f0': f0, 'AP': AP,
                'phn': per_frame_phn
            }
            with open(featpath, 'wb') as f:
                pickle.dump(feat, f)
            all_meta.append((f_id, text))

    # for metas
    with open(meta_path, 'w') as f:
        for meta in sorted(all_meta):
            f.write("{}|{}\n".format(meta[0], meta[1]))

    return

if __name__ == '__main__':
    main()

'''
### deprecated ###
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
            p_s_t = int(float(phn_bound.start_time) * ap.sr) - w_s_t
            p_e_t = int(float(phn_bound.end_time) * ap.sr) - w_s_t
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
