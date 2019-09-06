# Foreign Accent Conversion by Synthesizing Speech from Phonetic Posteriorgrams (accepted to Interspeech'19) 

This repository hosts the code we used to
prepare our interspeech'19 paper titled "[Foreign Accent Conversion by Synthesizing Speech from Phonetic Posteriorgrams](https://psi.engr.tamu.edu/wp-content/uploads/2019/07/zhao2019interspeech.pdf)"

### Install

This project uses `conda` to manage all the dependencies, you should install [anaconda](https://anaconda.org/) if you have not done so. 

```bash
# Clone the repo
git clone https://github.com/guanlongzhao/fac-via-ppg.git
cd $PROJECT_ROOT_DIR

# Install dependencies
conda env create -f environment.yml

# Activate the installed environment
conda activate ppg-speech

# Compile protocol buffer
protoc -I=src/common --python_out=src/common src/common/data_utterance.proto

# Include src in your PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT_DIR/src:$PYTHONPATH
```

If `conda` complains that some packages are missing, it is very likely that you can find a similar version of that package on anaconda's archive.

### Run unit tests

```bash
cd test

# Remember to make this script executable
./run_coverage.sh
```

This only does a few sanity checks, don't worry if the test coverage looks low :)

### Train PPG-to-Mel model
Change default parameters in `src/common/hparams.py:create_hparams()`.
The training and validation data should be specified in text files, see `data/filelists` for examples.

```bash
cd src/script
python train_ppg2mel.py
```
The `FP16` mode will not work, unfortunately.

### Train WaveGlow model
Change the default parameters in `src/waveglow/config.json`. The training data should be specified in the same manner as the PPG-to-Mel model.

```bash
cd src/script
python train_waveglow.py
```

### View training progress
You should find a dir `log` in all of your output dirs, that is the `LOG_DIR` you should use below.

```bash
tensorboard --logdir=${LOG_DIR}
```

### Generate speech synthesis
Use `src/script/generate_synthesis.py`, you can find pre-trained models in the [Links](#Links) section.

```bash
generate_synthesis.py [-h] --ppg2mel_model PPG2MEL_MODEL
                           --waveglow_model WAVEGLOW_MODEL
                           --teacher_utterance_path TEACHER_UTTERANCE_PATH
                           --output_dir OUTPUT_DIR
```

### Links

- Syntheses and pre-trained models: [link](https://drive.google.com/file/d/1nye-CAGyz3diM5Q80s0iuBYgcIL_cqrs/view?usp=sharing)
- Training data (L2-ARCTIC recordings after noise removal): [link](https://drive.google.com/file/d/1WnBHAfjEKdFTBDv5D6DxRnlcvfiODBgy/view?usp=sharing)
- Demo: [link](https://guanlongzhao.github.io/demo/fac-via-ppg)
