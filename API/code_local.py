from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import shutil
from donut_module.donut import initialize_processor, load_model, run_prediction
from ocr_module.ocr_transform import add_text, remove_text
from year_genre_module.year_genre_predicrion import predict_year, predict_genre
import pdb
from PIL import Image

# SELECT IMAGE TO PROCESS
fetch_dir = Path("../raw_data/img_received")
filename = 'tt0021148'
image = fetch_dir.joinpath(f'{filename}.jpg')
image = Image.open(image).convert('RGB')

# PREDICT YEAR AND GENRE
year = predict_year(image) # ==> array([1978.9]) <class 'numpy.ndarray'>
genre = predict_genre(image) # ==> ['Comedy', 'Drama'] <class 'list'>

# EXTRACT TITLE AND ACTORS
processor = initialize_processor()
model = load_model()
prediction = run_prediction(image, model, processor)

actors = " ".join(prediction['Actors']) # ==> 'Robert De Niro Eden Barkin Ellen Barkin' <class 'str'>

# REMOVE TEXT FROM IMAGE
new_image, title_words, title_location, actor_words, actor_location = remove_text(image, prediction['Title'], actors) # prediction['Title'] = "This Boy's Life" <class 'str'>

print('new_image', new_image) # ==> <class 'PIL.Image.Image'>
print('title_words', title_words) # ==> ['this', "boy's", 'life'] <class 'list'>
print('title_location', title_location) # ==> <class 'list'>
print('actor_words', actor_words) # ==> ['robert', 'de', 'niro', 'eden', 'barkin', 'ellen', 'barkin'] <class 'list'>
print('actor_location', actor_location) # ==> <class 'list'>

# ADD TEXT BACK TO IMAGE
new_image_ocr = add_text(new_image, title_words, title_location, actor_words, actor_location)
# print_words(image, text_list, bbox_text, font_style)

save_dir = Path("../raw_data/img_received")
file_path = save_dir / 'test_new.jpg'
new_image_ocr.save(file_path, format=None)
