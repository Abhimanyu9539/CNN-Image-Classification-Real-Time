from src.ML_pipeline import  Utils
import tensorflow as tf
from tensorflow import keras
from keras import  layers
from keras.models import Sequential

# Function to train Ml model

def train(model, train_ds, val_ds):
    epochs = 30
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=epochs)
    return model

# function to initiate the model and training data

def fit(train_ds, val_ds, class_names):
    num_classes = len(class_names)

    data_augmentaion = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(Utils.img_width,Utils.img_height,3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom((0.1))
        ]
    )

    model = Sequential([
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ['accuracy']
                  )


    model = train(model, train_ds, val_ds)
    print(model.summary())
    return model