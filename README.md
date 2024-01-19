# sudo-fix-dysarthria

Automatic speech recognition for people with dysarthria

[Built for Apple Silicon](https://developer.apple.com/metal/pytorch/), so might need some adjustments to work cross platform

## Installation

- Clone the repository: `git clone https://github.com/jmaczan/asr-dysarthria`
- Navigate to the project directory: `cd asr-dysarthria`
- Create a virtual environment: `python -m venv venv`
- Activate the virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- install git-lfs, for example: `apt install git-lfs`

## Notes

Metal Performance Shaders didn't work for me when installed with conda. Plain pip install works fine though

If you install another library, reflect it in requirements.txt by running `pip freeze > requirements.txt`

## Resources

Notebook:

https://colab.research.google.com/drive/1-Alt90u9nKQCS3ndr2FjpBuEZlmJwNWL

code:

https://huggingface.co/blog/fine-tune-wav2vec2-english

data:

http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html

dev dataset:

https://huggingface.co/datasets/jmaczan/TORGO-very-small

other resources:

https://pytorch.org/audio/stable/tutorials/speech_recognition_pipeline_tutorial.html

https://huggingface.co/docs/datasets/v2.16.1/audio_dataset - creating dataset

https://distill.pub/2017/ctc/
