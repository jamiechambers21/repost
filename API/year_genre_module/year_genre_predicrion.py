import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

GENRES = [
  'Family',
  'Action',
  'Comedy',
  'Thriller',
  'Romance',
  'Crime',
  'Sci-Fi',
  'Drama'
]

KNN_MODEL_YEAR_PATH = 'year_genre_module/model/knnpickle_years'
KNN_MODEL_GNERE_PATH = 'year_genre_module/model/knnpickle_genre'
AUTOENCODER = 'year_genre_module/model/autoencoder'
KNN_GENRE = "year_genre_module/model/genre_predict.npy"

def get_genres(indices):
  return [GENRES[i] for i in indices]

def build_multihot(indices, size=8):
  a = np.zeros((8,))
  a[indices] = 1
  return a

def predict_genres(new_X, n_genres, knn, y_genres_ref):
  X = new_X if new_X.ndim == 2 else new_X.reshape(1, -1) # if it's a single obs, reshape to fit sklearn model
  _, indices = knn.kneighbors(X)
  # given their indices, goes grabbing them
  neighbors_labels = y_genres_ref[indices] # axis 0, 1, 2: 0->each tested poster; 1->its neighbors; 2->genres

  # for each tested poster, the sum of genres of its neighbors
  neighbors_added_genres = neighbors_labels.sum(axis=1)
  genres_indices = np.argpartition(neighbors_added_genres, -n_genres)[:, -n_genres:]
  return genres_indices

def embed_images(image):
    #Load encoder
    autoencoder = models.load_model(AUTOENCODER)
    encoder = autoencoder.get_layer('encoder')

    img_np = np.expand_dims(image, axis=0)
    img_np = img_np.astype('float32') / 255.0
    img_tf = tf.convert_to_tensor(img_np)
    img_resized = tf.image.resize(img_tf, (528, 352))
    image_embeddings = encoder.predict(img_resized)
    img_emb = image_embeddings[0]
    return img_emb


def predict_year(image): #TODO: Put in embed images
    knn_model_year = pickle.load(open(KNN_MODEL_YEAR_PATH, 'rb'))
    image = embed_images(image)
    X = image if image.ndim == 2 else image.reshape(1, -1) # if it's a single obs, reshape to fit sklearn model

    return knn_model_year.predict(X)


def predict_genre(image):
    knn_model_genre = pickle.load(open(KNN_MODEL_GNERE_PATH, 'rb'))
    yg_train = np.load(KNN_GENRE)
    image = embed_images(image)
    indices = predict_genres(image, 2, knn_model_genre, yg_train)[0]
    return get_genres(indices)
