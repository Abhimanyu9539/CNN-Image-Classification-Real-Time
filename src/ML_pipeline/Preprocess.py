import tensorflow as tf
from src.ML_pipeline import Utils

# Function to cache the data for tensorflow

def cache_data(train_ds, valid_ds):
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, valid_ds


# Fucntion to calll dependent function

def apply(data_dir):
    print("Preprocessing started...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split = 0.2,
        subset = "training",
        seed = 123,
        image_size= (Utils.img_height, Utils.img_width),
        batch_size = Utils.batch_size
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(Utils.img_height,Utils.img_width),
        batch_size=Utils.batch_size
    )

    class_names = train_ds.class_names
    0
    print("Class Name : ", class_names)
    print("Data Loading completed..")

    train_ds, valid_ds = cache_data(train_ds, valid_ds)

    print("Preprocessing completed ...")
    return  train_ds, valid_ds, class_names