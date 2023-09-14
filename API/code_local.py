from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import shutil
from donut_module.donut import initialize_processor, load_model, run_prediction
from ocr_module.ocr_transform import add_text, remove_text
from year_genre_module.year_genre_predicrion import predict_year, predict_genre
from stable_dif.stable_dif import stable_diff
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

print('\n\n\n\nExtracted title, actors and predicted genre and year are: \n', f'Title: {prediction["Title"]}\n',
      f'Actors: {actors}\n', f'Year: {year[0]}\n', f'Genre: {genre}\n')

inp = input('If you have an NVIDIA graphics card, type "c" to continue with the stable difusion step (generate a new poster!), \nif not, refer to the google colab link on the ReadMe and hit "n" to skip the stable diffusion step: ')
if inp == 'c':
    inp = input('Do you wish to change the decade and genre? (Y/N) : ')
    if inp == 'Y':
        print(f'Current decade and genre are: {year} and {genre}')
        year = [input('Please enter the decade: ')]
        genre = [input('Please enter the genre: ')]
        stable_diff_img = stable_diff(image = new_image, genre = genre[0], decade = year[0], title = prediction['Title'])
        new_image_ocr = add_text(stable_diff_img, title_words, title_location, actor_words, actor_location)
    else:
        stable_diff_img = stable_diff(image = new_image, genre = genre[0], decade = year[0], title = prediction['Title'])
        new_image_ocr = add_text(stable_diff_img, title_words, title_location, actor_words, actor_location)
else:
    new_image_ocr = add_text(new_image, title_words, title_location, actor_words, actor_location)

# print_words(image, text_list, bbox_text, font_style)
save_dir = Path("../raw_data/img_received")
file_path = save_dir / 'test_20_new.jpg'
new_image_ocr.save(file_path, format=None)

new_image_ocr.show()
