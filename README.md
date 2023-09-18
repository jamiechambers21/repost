# RePost
Extract information from movie posters and create new ones 
Jamie Chambers, Fernando Ordorica Esquivel, Flora Von Waldow

## Requirement
The code is tested on Ubuntu with Nvidia GPUs and CUDA installed. Python>=3.6 is required to run the code. Computer must have a minium 6Gb of VRAM, if not please check out the notebook

## Installation
The following packages are required

! pip install keras_ocr
! nvidia-smi
! pip install diffusers==0.8.0 transformers ftfy accelerate # Had to change form 0.3.0 -> 0.8.0
! pip install -qq "ipywidgets>=7,<8"
! pip install transformers

## 🚀 How to use?
Simply load an image from your drive or by url and run the notebook to see the results

1. Donut (Extracting Text from Image)

2.Predict Year and Genre (Autoencoder+KNN)

3. Preparing the Poster (OCR)

4. Stable Diffusion

5. Final Image Processing (Reintroduce the text)

Donut
Document understanding transformer
- Identify text in images
- Extract the movie title and the actors from the poster
![Captura de pantalla 2023-09-08 164736](https://github.com/jamiechambers21/repost/assets/59603715/9e60dba9-082b-41b0-9f33-7d3d15f7fbb7)

Predict Year and Genre
AutoEncoder and k-Nearest Neighbors
- Employ an autoencoder to find meaningful patterns in the poster (unsupervised learning)
- Relate patterns with a KNN algorithm. Similar posters are also close to the computers eyes!

Preparing the Poster
OCR - Optical Character Recognition
- Used to locate where word boxes are in the poster
- Remove text from image
![Captura de pantalla 2023-09-08 171232](https://github.com/jamiechambers21/repost/assets/59603715/696fd0b6-fe07-479d-90a2-3ff92e0ba2ee)

Stable Diffusion
- Text-to-Image Generator
- Image-to-Image Generator
- Prompt the model to change the poster's genre and decade
![Captura de pantalla 2023-09-08 171217](https://github.com/jamiechambers21/repost/assets/59603715/61200187-cd5e-43c5-bad0-76284bd5f87a)

Final Image Processing
- Using information extracted (text, location generated image)
- Adding text on the generated poster at location where the text is on the original
![Captura de pantalla 2023-09-08 172755](https://github.com/jamiechambers21/repost/assets/59603715/b82b3a1b-d0b0-4954-b331-3178c6018c9f)
