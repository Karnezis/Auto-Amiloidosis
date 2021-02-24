import autokeras as ak
import tensorflow as tf
import os

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
local_file_path = tf.keras.utils.get_file(origin=dataset_url, 
                                          fname='image_data', 
                                          extract=True)
# The file is extracted in the same directory as the downloaded file.
local_dir_path = os.path.dirname(local_file_path)
# After check mannually, we know the extracted data is in 'flower_photos'.
data_dir = os.path.join(local_dir_path, 'flower_photos')
print(data_dir)

batch_size = 32
img_height = 180
img_width = 180

train_data = ak.image_dataset_from_directory(
    data_dir,
    # Use 20% data as testing data.
    validation_split=0.2,
    subset="training",
    # Set seed to ensure the same split when loading testing data.
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_data = ak.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

clf = ak.ImageClassifier(overwrite=True, max_trials=1)
clf.fit(train_data, epochs=1)
print(clf.evaluate(test_data))