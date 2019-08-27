# Foreign Accent Conversion by Synthesizing Speech from Phonetic Posteriorgrams (accepted to Interspeech'19)

**THIS IS A PRERELEASE; MORE DOCUMENTATIONS AND CODE-HOUSEKEEPING ON THE WAY**

PPG->Speech conversion code. This branch hosts the original code we used to
prepare our interspeech'19 paper titled "Foreign Accent Conversion by Synthesizing Speech from Phonetic Posteriorgrams"

### Install

```bash
# Dependencies
conda env create -f environment.yml

# Compile protocol buffer
protoc -I=src/common --python_out=src/common src/common/data_utterance.proto
```

### Run unit tests

```bash
cd test
./run_coverage.sh
```

### Train
Change default parameters in `hparams.py`
```bash
cd src/script
python train.py
```

### View progress
```bash
tensorboard --logdir=${LOG_DIR}
```

### Links

- Syntheses and pretraind models: [link](https://drive.google.com/file/d/1nye-CAGyz3diM5Q80s0iuBYgcIL_cqrs/view?usp=sharing)
- Additional content in `data`: [link](https://drive.google.com/file/d/1aX8-df-5fTa-wrPwU8CyzIML1_u7ZDIN/view?usp=sharing)
- Training data (L2-ARCTIC recordings after noise removal): [link](https://drive.google.com/file/d/1WnBHAfjEKdFTBDv5D6DxRnlcvfiODBgy/view?usp=sharing)
- Demo: [link](https://guanlongzhao.github.io/demo/fac-via-ppg)