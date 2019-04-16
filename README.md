# Non-parallel-rhythm-flexible-VC
PyTorch implementation of: 
[Rhythm-Flexible Voice Conversion without Parallel Data Using Cycle-GAN over Phoneme Posteriorgram Sequences](https://arxiv.org/abs/1808.03113)

# Data Preprocess
1. Download and decompress VCTK corpus
2. Put text file and audio file under same dir, run `rename.sh`
3. Run align_VCTK.sh to get aligned result
4. Set path info in config/config.yaml
5. Run `preprocess.py` to generate acoustic features with corresponding phone label

# Network

# Notes
1. Perhaps not to use dataloader at inference time?
2. phoneme 'spn' means Unknown in MFA, so currently map it with 'sp' to id 0 as well.
