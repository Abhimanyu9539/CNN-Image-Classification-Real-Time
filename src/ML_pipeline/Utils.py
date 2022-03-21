import keras.models
import tensorflow

batch_size = 32
img_height = 180
img_width = 180

def save_model(model):
    model.save("../output/cnn-model.h5")
    return True


def load_model(model_path):
    model = None
    try:
        model = keras.models.load_model(model_path)
    except:
        print("Please enter valid model path")
        exit(0)

    return model