import mne
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.applications import VGG16


def extract_spectrogram(edf_file: str):
    raw_edf = mne.io.read_raw_edf(f'{edf_file}', preload=True)
    raw_edf.filter(l_freq=1, h_freq=None)
    data_channels = raw_edf.get_data()
    data_channel = data_channels[0]

    plt.specgram(data_channel, Fs=raw_edf.info['sfreq'], cmap='viridis', aspect='auto', NFFT=256, noverlap=128)
    plt.axis('off')
    plt.savefig(f'{edf_file}.png', bbox_inches='tight', pad_inches=0, transparent=True)

    return load_img(f"{edf_file}.png", target_size=(224, 224))

def inference(model, file_path: str, image:bool):
    if image:
        img = load_img(f"upload/{file_path}", target_size=(224, 224))
    else:
        img = extract_spectrogram(f"upload/{file_path}")

    #preprocess the image
    my_image = img_to_array(img)
    my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
    my_image = preprocess_input(my_image)

    #make the prediction
    prediction = model.predict(my_image)
    labels = ["Rest", "Thinking of moving hand"]
    return labels[np.argmax(prediction)]

def load_model():
    pre_trained_model = VGG16(weights='imagenet',
                              include_top=False,
                              input_shape=(224, 224, 3))

    for layer in pre_trained_model.layers:
        layer.trainable = False

    model = keras.Sequential([
        pre_trained_model,
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2, activation='softmax')
    ])

    path_model = 'model/model_simple.h5'
    model.load_weights(path_model)
    return model