import os
from transformers import AutoProcessor, AutoModelForCTC

cache_dir = "cached_model"


def load_or_create_model(
    model_name: str = "jmaczan/wav2vec2-large-xls-r-300m-dysarthria",
    to_eval: bool = True,
):
    if os.path.exists(cache_dir):
        print("Loading model from cache...")
        processor = AutoProcessor.from_pretrained(cache_dir)
        model = AutoModelForCTC.from_pretrained(cache_dir)
    else:
        print("Downloading and caching model...")
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCTC.from_pretrained(model_name)
        processor.save_pretrained(cache_dir)
        model.save_pretrained(cache_dir)
    if to_eval:
        model.eval()

    return processor, model
