#DEPENDENCIES
import os
import json
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from datasets import load_dataset, load_from_disk
import random
from transformers import DonutProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
import torch
from PIL import Image
from huggingface_hub import HfFolder
import accelerate
import re
from tqdm import tqdm
import torchvision
import torchvision.transforms as T
from difflib import SequenceMatcher
import pdb

#INITIALIZE SEED
set_seed(42)

# DEFINE PATHS

cache_dir = 'code/jamiechambers21/repost/API'
processor_path = 'donut_module/processor'
model_path = 'donut_module/model'

def initialize_processor():
    # SPECIAL TOKENS

    processor = DonutProcessor.from_pretrained(processor_path, local_files_only=True, cache_dir=cache_dir)

    return processor


def load_model():
    # LOAD MODEL

    model = VisionEncoderDecoderModel.from_pretrained(model_path, local_files_only=True, cache_dir=cache_dir)

    return model


def run_prediction(sample, model, processor):

    # prepare inputs
    pixel_values = processor(sample, random_padding=True, return_tensors="pt").pixel_values

    # UserWarning: `num_beams` is set to 1. However, `early_stopping` is set to `True` --
    # this flag is only used in beam-based generation modes.
    # You should set `num_beams>1` or unset `early_stopping`.

    # UserWarning: `num_beams` is set to 1. However, `early_stopping` is set to `True` -- this flag is only used in beam-based generation modes. You should set `num_beams>1` or unset `early_stopping`.

    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # asign device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    # run inference
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        # early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # process output
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = processor.token2json(prediction)

    return prediction


# # PROCESS IMAGE
# # PSEUDOCODE:
#     image = type['PIL.JpegImagePlugin.JpegImageFile'].convert('RGB') # --> <class 'PIL.Image.Image'>
#     pixel_values = processor(
#             sample["image"], random_padding=split == "train", return_tensors="pt"
#         ).pixel_values.squeeze()

# CONVERT FROM PIL.JpegImagePlugin.JpegImageFile TO RGB: <class 'PIL.Image.Image'>

# MAYBE NEED TO RESIZE?

# TURN IMAGE INTO torch.Tensor
