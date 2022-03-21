import pathlib
import subprocess
import tensorflow as tf
import src.ML_pipeline
from src.ML_pipeline import Train_Model, Predict, Preprocess, Utils
from src.ML_pipeline.Preprocess import apply
from src.ML_pipeline.Utils import  save_model,load_model

val = int(input("Train - 0\nPredict - 1\nDeploy - 2\nEnter your value :"))

if val==0:
    data_dir = pathlib.Path('../input/Training_data/')
    image_count = len(list(data_dir.glob('*/*')))
    print("Number of images for training ",image_count)

    train_ds, val_ds, class_name  = apply(data_dir)
    ml_model = Train_Model.fit(train_ds, val_ds, class_name)
    model_path = save_model(ml_model)
    print("Model saved in:","../output/cnn-model")

elif val==1:
    model_path = "../output/cnn-model.h5"
    ml_model = load_model(model_path)
    test_data_dir = pathlib.Path("../input/Testing_Data/")
    image_count = len(list(test_data_dir.glob('*/*')))
    print("Number of images for testing: ", image_count )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        seed=123,
        image_size=(Utils.img_height, Utils.img_width),
        batch_size = Utils.batch_size
    )

    prediction = ml_model.predict(test_ds)
    print(prediction)
    print(ml_model.evaluate(test_ds))

else:
    # For prod deployment
    '''process = subprocess.Popen(['sh', 'ML_Pipeline/wsgi.sh'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )'''

    # For dev deployment
    process = subprocess.Popen(['python', 'ML_Pipeline/deploy.py'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True
                               )

    for stdout_line in process.stdout:
        print(stdout_line)

    stdout, stderr = process.communicate()
    print(stdout, stderr)
