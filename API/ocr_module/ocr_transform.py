import cv2
import numpy as np
import pandas as pd
import keras_ocr
import math
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
from difflib import SequenceMatcher
import pdb

pipeline = keras_ocr.pipeline.Pipeline()


# Delete this file/folder info
# df = pd.read_csv("movies_val_years.csv")
# images_path = Path("raw_data/photos/movies_val_resized/")

"""Take in an image, locate text, make df of text locations,
    make new image without the text"""
def process_image_and_inpaint(image, pipeline):
    # Resize image to the size we will work with
    image = image.resize((352, 528)) # Look into this: Should be fixing all our problems

    img = np.asarray(image)

    prediction_groups = pipeline.recognize([img])

    texts = []
    bboxes = []

    for result in prediction_groups[0]:
        texts.append(result[0])
        bboxes.append(result[1])

    text_df = pd.DataFrame({'text': texts, 'bbox': bboxes})

    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1]
        x2, y2 = box[1][2]
        x3, y3 = box[1][3]

        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mid1 = midpoint(x0, y0, x3, y3)

        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

    return text_df, inpainted_img

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return (x_mid, y_mid)

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Convert np array back to image
def np_to_img(np_image):
    img = Image.fromarray(np_image, "RGB")
    # Resize image as Stable Diffusion needs an image %8, Org:525, 350
    img = img.resize((352, 528))
    return img

def words_location(text, text_df):
    text_list = text.lower().split()
    bbox_text = []

    for text_word in text_list:
        if text_word in list(text_df.text):
            bbox_text.append([x.round() for x in text_df[text_df.text == text_word].bbox][0])
        else :
            for word in list(text_df.text):
                if similar(text_word,word) > .75:
                    bbox_text.append([x.round() for x in text_df[text_df.text == word].bbox][0])
                    break

    return text_list, bbox_text


# Put the words back on the image in the same place they were before
def print_words(image, text_list, bbox_text, font_style):

    draw = ImageDraw.Draw(image)

    # Making sure the colour is different from the background
    # Select first bbox of first word
    box = bbox_text[0]
    cor_left = min(np.floor(box[0][0]), np.floor(box[2][0]))
    cor_upper= min(np.floor(box[0][1]), np.floor(box[2][1]))
    cor_right= max(np.floor(box[0][0]), np.floor(box[2][0]))
    cor_lower= max(np.floor(box[0][1]), np.floor(box[2][1]))

    # Crop where text should be, find most common colour and choose the opposite
    crop_img = image.crop((cor_left, cor_upper, cor_right, cor_lower))
    most_common = sorted(crop_img.getcolors(maxcolors=2**16), key=lambda t: t[0], reverse=True)[0]
    fill=(255-most_common[1][0], 255-most_common[1][1], 255-most_common[1][2])

    for i, bbox in enumerate(bbox_text):
        a, b, = bbox[0]
        text = text_list[i]
        font = ImageFont.truetype(f"../raw_data/font_files/{font_style}.ttf", 1)
        font_len = 0
        font_size = 0

        # Making font bigger until text is the same size as before
        while font_len <= (bbox[1][0] - bbox[0][0]):
            font = ImageFont.truetype(f"../raw_data/font_files/{font_style}.ttf", font_size+1)
            font_size = font_size + 1
            font_len = font.getlength(text)

        font = ImageFont.truetype(f"../raw_data/font_files/{font_style}.ttf", font_size)
        draw.text((a, b), text=text, font=font , fill=fill)

    return image


def remove_text(image, title, actors, pipeline=pipeline):

    # Get df of text locations and image without text
    text_df, img_text_removed = process_image_and_inpaint(image, pipeline)

    # Image is in numpy format, convert
    image = np_to_img(img_text_removed)

    # Get words and locaions of Title and Actors
    title_words, title_location = words_location(title, text_df)
    actor_words, actor_location = words_location(actors, text_df)

    return image, title_words, title_location, actor_words, actor_location


def add_text(image, title_words, title_location, actor_words, actor_location):

    # Choose font style #TODO: change to Drama, Comedy...
    font_style = ["LuckiestGuy", "Mistral", "Serpentine"]

    # Put text for Title and Actors on the image
    img_w_title = print_words(image, title_words, title_location, font_style[0])
    image_w_all = print_words(img_w_title, actor_words, actor_location, font_style[0])

    return image_w_all
