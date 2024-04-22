"""Usage: inference.py [-vh] NAME [-e EPOCH]

Test the gloss predictor
Arguments:
  NAME        experiment name: e.g., s_100_only, s_100_bsdf, s_20_imagestats...
Options:
  -h, --help
  -v                        verbose mode: it prints the predictions for every image in every test set and saves latent vectors
                                if no verbose, just MAE and Spearman and Pearson correlation for our test dataset and serrano testsetB are printed.
  -e EPOCH --epoch=EPOCH    epoch of the checkpoint to test [default: 35]
"""
from docopt import docopt

import os
from PIL import Image
import numpy as np
import pandas as pd
from scipy import stats

import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16

image_size = 512
batch_size = 1

arguments = docopt(__doc__)

name_experiment = arguments['NAME']
prefix = 'weights_'+arguments['--epoch']

path_experiment = '../models/'+name_experiment+'/'
path_datasets = '../data/'

file_weights = [file for file in os.listdir(path_experiment) if file.startswith(prefix)][0]

class custom_preprocessing(tf.keras.layers.Layer):
    def __init__(self):
        super(custom_preprocessing, self).__init__()
    
    def call(self, inputs):
        inputs = inputs / 255.0
        return inputs


def adapt_vgg16() -> Model:
    """This code uses adapts the most up-to-date version of EfficientNet with NoisyStudent weights to a regression
    problem. Most of this code is adapted from the official keras documentation.

    Returns
    -------
    Model
        The keras model.
    """
    input_tensor = layers.Input(
        shape=(image_size, image_size, 3)
    )
    preprocessed_inputs = custom_preprocessing()(input_tensor)
    model_vgg = VGG16(weights="imagenet", include_top=False)

    # Do not freeze the pretrained weights
    model_vgg.trainable = True

    x = model_vgg(preprocessed_inputs)
    # Rebuild top
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, name="fc1", activation="relu")(x)
    x = layers.Dense(20, name="latent", activation="relu")(x)
    outputs = layers.Dense(1, name="pred", activation="sigmoid")(x)

    # Compile
    model = keras.Model(input_tensor, outputs, name="VGG16")

    return model


def open_images_from_df(inference_folder: str, labels_df) -> np.ndarray:
    """Loads images from a folder and prepare them for inferencing.

    Parameters
    ----------
    inference_folder : str
        Location of images for inferencing.
    labels_df: Pandas Dataframe


    Returns
    -------
    np.ndarray
        List of images as numpy arrays transformed to fit the network input specs.
    """
    df = labels_df
    images = []
    labels = []
    images_names = []
    for img in df['image']:
        img_location = os.path.join(inference_folder, img)  # create full path to image
        if (os.path.isfile(img_location)): # only if image exists
            labels.append(df.loc[df['image']==img]['glossiness'].values[0])
            images_names.append(img)

            with Image.open(img_location) as imag:  # open image with pillow

                imag = imag.resize((image_size,image_size), Image.ANTIALIAS)
                imag = np.array(imag)
                imag = imag[:, :, :3]
                imag = np.expand_dims(imag, axis=0)  # add 0 dimension to fit input shape of network

            images.append(imag)

        else:
            print('WARNING: Image', img_location, 'does not exist')
    images_array = np.vstack(images)  # combine images efficiently to a numpy array
    labels_array = np.array(labels)
    return images_array, labels_array, images_names

############################################# INFERENCE CODE ################################################
print('############################ INFERENCE RESULTS #####################################')
print('MODEL: ', name_experiment)
print('Checkpoint: ', file_weights)

model = adapt_vgg16()

model.load_weights(path_experiment + file_weights)

extractor = keras.Model(inputs=model.inputs, outputs=[model.layers[-2].output, model.layers[-1].output])

# PREDICT WITH TEST SET B SERRANO 21
print(' -- TEST SET B (Serrano et al. 2021)')
testsetb = pd.read_csv(path_datasets+"serrano21_setB_eval/setB_removed_sphere.csv")
testsetb["image_location"] = (path_datasets+"serrano21_setB_eval/" + testsetb["image"])

# add glossiness in range [0-1]
testsetb["glossiness_01"] = (((testsetb["glossiness"] - 1.0)/6.0))
testsetb_generator = ImageDataGenerator()
testsetb_generator = testsetb_generator.flow_from_dataframe(
        dataframe=testsetb,
        x_col="image_location",
        y_col="glossiness_01",
        class_mode="raw",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

images, labels, images_names = open_images_from_df(path_datasets+"serrano21_setB_eval/", testsetb)

latents, predictions = extractor.predict(images)

if arguments['-v']:
    if not os.path.exists(path_experiment+"latents/"):
        os.makedirs(path_experiment+"latents/")
    print('Latents TEST SET B shape', latents.shape)
    np.save(path_experiment +'latents/testsetb_latent_' + name_experiment + '.npy', latents)
    np.save(path_experiment +'latents/testsetb_labels_' + name_experiment + '.npy', labels)

aes = []
aes_01 = []
accuracies = []
list_all_labels = []
list_all_predictions = []
for image_name, prediction, label in zip(images_names, predictions, labels):
    prediction_17 = prediction * 6 + 1
    ae = np.absolute(prediction_17 - label)
    aes.append(ae)
    label_01 = (label - 1.0) / 6.0
    ae_01 = np.absolute(prediction - label_01)
    list_all_predictions.append(prediction[0])
    list_all_labels.append(label_01)
    aes_01.append(ae_01)
    accuracies.append(int(np.absolute(label - np.round(prediction_17)) < 1))
    if arguments['-v']:
        print(image_name, prediction[0], label_01, prediction_17[0], label)

if arguments['-v']:
    print('MAE [1-7]: ', np.mean(aes), ' std: ', np.std(aes))
    print('MAE [0-1]: ', np.mean(aes_01), ' std: ', np.std(aes_01))
    print('ACCURACY: ', np.mean(accuracies), ' std: ', np.std(accuracies))
else:
    print('MAE [0-1]: ', np.mean(aes_01))

# Add Spearman and Pearson correlations
res_asymptotic = stats.spearmanr(list_all_predictions, list_all_labels)
print('Spearman correlation', res_asymptotic)
res_pearson = stats.pearsonr(np.array(list_all_predictions), np.array(list_all_labels))
print('Pearson correlation', res_pearson)

# PREDICT WITH NEW TEST SET
print(' -- OUR NEW TEST DATASET')
new_testset = pd.read_csv(path_datasets+"new_test_png/test_set_labels.csv")
new_testset["image_location"] = (path_datasets+"new_test_png/" + new_testset["image"])

# add glossiness in range [0-1]
new_testset["glossiness"] = new_testset["median"]
new_testset["glossiness_01"] = (((new_testset["median"] - 1.0)/6.0))
new_testset_generator = ImageDataGenerator()
new_testset_generator = new_testset_generator.flow_from_dataframe(
        dataframe=new_testset,
        x_col="image_location",
        y_col="glossiness_01",
        class_mode="raw",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

images, labels, images_names = open_images_from_df(path_new_testset, new_testset)

latents, predictions = extractor.predict(images)

if arguments['-v']:
    if not os.path.exists(path_experiment+"latents/"):
        os.makedirs(path_experiment+"latents/")
    print('Latents NEW TEST SET shape', latents.shape)
    np.save(path_experiment +'latents/new_testset_latent_' + name_experiment + '.npy', latents)
    np.save(path_experiment +'latents/new_testset_labels_' + name_experiment + '.npy', labels)

aes = []
aes_01 = []
accuracies = []
offset_gt = []
list_all_predictions = []
list_all_labels = []
for image_name, prediction, label in zip(images_names, predictions, labels):
    prediction_17 = prediction * 6 + 1
    ae = np.absolute(prediction_17 - label)
    aes.append(ae)
    if label == 7:
        offset_gt.append(prediction_17)
    label_01 = (label - 1.0) / 6.0
    ae_01 = np.absolute(prediction - label_01)
    aes_01.append(ae_01)
    list_all_predictions.append(prediction[0])
    list_all_labels.append(label_01)
    accuracies.append(int(np.absolute(label - np.round(prediction_17)) < 1))
    if arguments['-v']:
        print(image_name, prediction[0], label_01, prediction_17[0], label)

if arguments['-v']:
    print('MAE [1-7]: ', np.mean(aes), ' std: ', np.std(aes))
    print('MAE [0-1]: ', np.mean(aes_01), ' std: ', np.std(aes_01))
    print('ACCURACY: ', np.mean(accuracies), ' std: ', np.std(accuracies))
else:
    print('MAE [0-1]: ', np.mean(aes_01))

# Add Spearman and Pearson correlations
res_asymptotic = stats.spearmanr(list_all_predictions, list_all_labels)
print('Spearman correlation', res_asymptotic)
res_pearson = stats.pearsonr(np.array(list_all_predictions), np.array(list_all_labels))
print('Pearson correlation', res_pearson)


# PREDICT WITH PERFECTLY SPECULAR PLANES
if arguments['-v']:
    print(' -- SPECULAR PLANES')
    specular_planes = pd.read_csv(path_datasets+"plane_images/plane_images.csv")
    specular_planes["image_location"] = (path_datasets+"plane_images/" + specular_planes["image"])

    # add glossiness in range [0-1]
    specular_planes["glossiness_01"] = (((specular_planes["glossiness"] - 1.0)/6.0))
    specular_planes_generator = ImageDataGenerator()
    specular_planes_generator = specular_planes_generator.flow_from_dataframe(
            dataframe=specular_planes,
            x_col="image_location",
            y_col="glossiness_01",
            class_mode="raw",
            target_size=(image_size, image_size),
            batch_size=batch_size,
            shuffle=False,
        )

    images, labels, images_names = open_images_from_df(path_datasets+"plane_images/", specular_planes)

    latents, predictions = extractor.predict(images)

    aes_01 = []
    list_all_labels = []
    list_all_predictions = []
    for image_name, prediction, label in zip(images_names, predictions, labels):
        prediction_17 = prediction * 6 + 1
        label_01 = (label - 1.0) / 6.0
        ae_01 = np.absolute(prediction - label_01)
        list_all_predictions.append(prediction[0])
        list_all_labels.append(label_01)
        aes_01.append(ae_01)
        print(image_name, prediction[0], label_01, prediction_17[0], label)

    print('MAE [0-1]: ', np.mean(aes_01))
