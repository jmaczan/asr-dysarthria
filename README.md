# 😺 `sudo fix --dysarthria`

Automatic speech recognition for people with dysarthria

## Training the model

`python src/train.py`

## Building the TORGO dataset

`python src/torgo_dataset_builder.py`

http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html

## Open questions

- How to obtain Nemours database of dysarthric speech? https://ieeexplore.ieee.org/document/608020

## Installation

- Clone the repository: `git clone https://github.com/jmaczan/asr-dysarthria`
- Navigate to the project directory: `cd asr-dysarthria`
- Create a virtual environment: `python -m venv venv`
- Activate the virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- install git-lfs, for example: `apt install git-lfs`

## Notes

(initally) [built for Apple Silicon](https://developer.apple.com/metal/pytorch/), so might need some adjustments to work cross platform

Metal Performance Shaders didn't work for me when installed with conda. Plain pip install works fine though

If you install another library, reflect it in requirements.txt by running `pip freeze > requirements.txt`

## Resources

papers:

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10225595

https://www.sciencedirect.com/science/article/pii/S2405959521000874

https://www.isca-speech.org/archive/pdfs/interspeech_2021/green21_interspeech.pdf

https://arxiv.org/pdf/2006.11477.pdf

https://arxiv.org/pdf/2211.00089.pdf

https://www.sciencedirect.com/science/article/abs/pii/S0957417423002981

Notebook:

https://colab.research.google.com/drive/1-Alt90u9nKQCS3ndr2FjpBuEZlmJwNWL

code:

https://huggingface.co/blog/fine-tune-wav2vec2-english

data:

http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html

dev dataset:

https://huggingface.co/datasets/jmaczan/TORGO-very-small

other resources:

https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/

https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html

https://huggingface.co/docs/datasets/v2.16.1/audio_dataset - creating dataset

https://distill.pub/2017/ctc/

https://ai.meta.com/blog/self-supervision-and-building-more-robust-speech-recognition-systems/
