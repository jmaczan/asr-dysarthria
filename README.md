# ASR Dysarthria

Automatic speech recognition for people with dysarthria

## Training

Use this Jupyter Notebook [wav2vec2-large-xls-r-300m-dysarthria-big-dataset.ipynb](wav2vec2-large-xls-r-300m-dysarthria-big-dataset.ipynb) to train your own model

## Deploying

Download and convert trained model (model.safetensors file)

```sh
mkdir models
python scripts/convert_model.py --url https://huggingface.co/jmaczan/wav2vec2-large-xls-r-300m-dysarthria-big-dataset/resolve/main/model.safetensors --output models
```

Serve it

```
cd web-app
python -m http.server
```

## Pretrained models

- [Recommended] Loss: 0.0864, Wer: 0.182 https://huggingface.co/jmaczan/wav2vec2-large-xls-r-300m-dysarthria-big-dataset
- Loss: 0.0615 Wer: 0.1764 https://huggingface.co/jmaczan/wav2vec2-large-xls-r-300m-dysarthria

## Datasets

- Uaspeech https://huggingface.co/datasets/Vinotha/uaspeechall
- TORGO https://huggingface.co/datasets/jmaczan/TORGO

## Description

The code here is based on Patrick von Platen's article and notebook https://huggingface.co/blog/fine-tune-xlsr-wav2vec2

## Resources

### Papers

https://ar5iv.labs.arxiv.org/html/2204.00770 (https://arxiv.org/abs/2204.00770)

https://www.isca-speech.org/archive/pdfs/interspeech_2022/baskar22b_interspeech.pdf

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10225595

https://www.sciencedirect.com/science/article/pii/S2405959521000874

https://www.isca-speech.org/archive/pdfs/interspeech_2021/green21_interspeech.pdf

https://arxiv.org/pdf/2006.11477.pdf

https://arxiv.org/pdf/2211.00089.pdf

https://www.sciencedirect.com/science/article/abs/pii/S0957417423002981

### Code

https://huggingface.co/blog/fine-tune-wav2vec2-english

### Data

http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html

### Dataset

#### Big

https://huggingface.co/datasets/jmaczan/TORGO

#### Small

https://huggingface.co/datasets/jmaczan/TORGO-very-small

### Others

https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/

https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html

https://huggingface.co/docs/datasets/v2.16.1/audio_dataset

https://distill.pub/2017/ctc/

https://ai.meta.com/blog/self-supervision-and-building-more-robust-speech-recognition-systems/

## License

MIT License

## Author

Jędrzej Paweł Maczan

https://huggingface.co/jmaczan | jed@maczan.pl | https://github.com/jmaczan
