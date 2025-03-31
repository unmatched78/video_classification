# Import necessary libraries
import tqdm
import random
import pathlib
import collections
import os
import cv2
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
import mediapy as media
from moviepy.editor import VideoFileClip

# Define helper functions for data processing
def get_class(fname):
    return fname.split('_')[-3]

def list_files_from_zip_url(zip_url):
    files = []
    with rz.RemoteZip(zip_url) as zip:
        for zip_info in zip.infolist():
            files.append(zip_info.filename)
    return files

def get_files_per_class(files):
    files_for_class = collections.defaultdict(list)
    for fname in files:
        class_name = get_class(fname)
        files_for_class[class_name].append(fname)
    return files_for_class

def select_subset_of_classes(files_for_class, classes, files_per_class):
    files_subset = dict()
    for class_name in classes:
        class_files = files_for_class[class_name]
        files_subset[class_name] = class_files[:files_per_class]
    return files_subset

def download_from_zip(zip_url, to_dir, file_names):
    with rz.RemoteZip(zip_url) as zip:
        for fn in tqdm.tqdm(file_names):
            class_name = get_class(fn)
            zip.extract(fn, str(to_dir / class_name))
            unzipped_file = to_dir / class_name / fn
            fn = pathlib.Path(fn).parts[-1]
            output_file = to_dir / class_name / fn
            unzipped_file.rename(output_file)

def split_class_lists(files_for_class, count):
    split_files = []
    remainder = {}
    for cls in files_for_class:
        split_files.extend(files_for_class[cls][:count])
        remainder[cls] = files_for_class[cls][count:]
    return split_files, remainder

def download_ucf_101_subset(zip_url, num_classes, splits, download_dir, classes_1):
    files = list_files_from_zip_url(zip_url)
    for f in files:
        path = os.path.normpath(f)
        tokens = path.split(os.sep)
        if len(tokens) <= 2:
            files.remove(f)
    files_for_class = get_files_per_class(files)
    classes = classes_1
    for cls in classes:
        random.shuffle(files_for_class[cls])
    files_for_class = {x: files_for_class[x] for x in classes}
    dirs = {}
    for split_name, split_count in splits.items():
        split_dir = download_dir / split_name
        split_files, files_for_class = split_class_lists(files_for_class, split_count)
        download_from_zip(zip_url, split_dir, split_files)
        dirs[split_name] = split_dir
    return dirs

def format_frames(frame, output_size):
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame

def frames_from_video_file(video_path, n_frames, output_size=(224, 224), frame_step=15):
    result = []
    src = cv2.VideoCapture(str(video_path))
    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    need_length = 1 + (n_frames - 1) * frame_step
    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)
    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))
    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]
    return result

class FrameGenerator:
    def __init__(self, path, n_frames, training=False):
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*.avi'))
        classes = [p.parent.name for p in video_paths]
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()
        pairs = list(zip(video_paths, classes))
        if self.training:
            random.shuffle(pairs)
        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames)
            label = self.class_ids_for_name[name]
            yield video_frames, label

# Define functions for model building and training
def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
    model = movinet_model.MovinetClassifier(
        backbone=backbone,
        num_classes=num_classes)
    model.build([batch_size, num_frames, resolution, resolution, 3])
    return model

def load_pretrained_movinet(model_id='a0'):
    tf.keras.backend.clear_session()
    backbone = movinet.Movinet(model_id=model_id)
    backbone.trainable = False
    model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
    model.build([None, None, None, None, 3])
    checkpoint_dir = f'movinet_{model_id}_base'
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model)
    status = checkpoint.restore(checkpoint_path)
    status.assert_existing_objects_matched()
    return model, backbone

# Download and prepare the dataset
URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'
NUM_CLASSES = 25
FILES_PER_CLASS = 100
classes_1 = ['Archery', 'BabyCrawling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'CliffDiving', 'CuttingInKitchen', 'Diving', 'Fencing', 'Hammering', 'HammerThrow', 'HighJump', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'LongJump', 'Mixing', 'MoppingFloor', 'PoleVault', 'Punch', 'SalsaSpin', 'Shotput', 'SumoWrestling', 'TaiChi', 'ThrowDiscus', 'TrampolineJumping']

download_dir = pathlib.Path('./UCF101_subset/')
subset_paths = download_ucf_101_subset(URL, num_classes=NUM_CLASSES, splits={"train": 30, "val": 10, "test": 10}, download_dir=download_dir, classes_1=classes_1)

batch_size = 8
num_frames = 8
output_signature = (tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], num_frames, training=True), output_signature=output_signature).batch(batch_size)
val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], num_frames), output_signature=output_signature).batch(batch_size)

# Load and build the model
model, backbone = load_pretrained_movinet()
model = build_classifier(batch_size, num_frames, 224, backbone, NUM_CLASSES)

# Compile and train the model
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

num_epochs = 2
results = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, validation_freq=1, verbose=1)
model.save('saved_model/my_model')

# Evaluate the model
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5)
    ax1.set_title('Loss')
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='test')
    ax1.set_ylabel('Loss')
    max_loss = max(history.history['loss'] + history.history['val_loss'])
    ax1.set_ylim([0, np.ceil(max_loss)])
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])
    ax2.set_title('Accuracy')
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='test')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])
    plt.savefig("loss.png")
    plt.show()

def get_actual_predicted_labels(model, dataset):
    actual = [labels for _, labels in dataset.unbatch()]
    predicted = model.predict(dataset)
    actual = tf.stack(actual, axis=0)
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)
    return actual, predicted

def plot_confusion_matrix(actual, predicted, labels):
    con_mat = tf.math.confusion_matrix(labels=actual, predictions=predicted).numpy()
    con_mat_df = pd.DataFrame(con_mat, index=labels, columns=labels)
    figure = plt.figure(figsize=(9, 9))
    sns.heatmap(con_mat_df, annot=True, fmt='g')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Confusion_matrix_test.png', bbox_inches='tight')
    plt.show()

model = tf.keras.models.load_model('saved_model/my_model')
test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], num_frames), output_signature=output_signature).batch(batch_size)

results = model.evaluate(test_ds, return_dict=True)
plot_history(results)

actual, predicted = get_actual_predicted_labels(model, test_ds)
plot_confusion_matrix(actual, predicted, classes_1)

# Inference on GIFs
def load_gif(file_path, image_size=(224, 224)):
    with tf.io.gfile.GFile(file_path, 'rb') as f:
        video = tf.io.decode_gif(f.read())
    video = tf.image.resize(video, image_size)
    video = tf.cast(video, tf.float32) / 255.
    return video

def get_top_k(probs, k=5, label_map=classes_1):
    top_predictions = tf.argsort(probs, axis=-1, direction='DESCENDING')[:k]
    top_labels = tf.gather(label_map, top_predictions, axis=-1)
    top_labels = [label.decode('utf8') for label in top_labels.numpy()]
    top_probs = tf.gather(probs, top_predictions, axis=-1).numpy()
    return tuple(zip(top_labels, top_probs))

def predict_top_k(model, video, k=5, label_map=classes_1):
    outputs = model.predict(video[tf.newaxis])[0]
    probs = tf.nn.softmax(outputs)
    return get_top_k(probs, k=k, label_map=label_map)

# Convert videos to GIFs for inference
directory = '/content/video_classification/UCF101_subset/test'
save_dir = '/content/video_classification/sample_gifs'
subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]

for i in range(5):
    subdir = random.choice(subdirectories)
    subdir_path = os.path.join(directory, subdir)
    files = [file for file in os.listdir(subdir_path) if file.endswith('.avi')]
    random_file = random.choice(files)
    input = directory + '/' + subdir + '/' + random_file
    clip = VideoFileClip(input)
    output = save_dir + '/' + random_file.split('.')[0] + '.gif'
    clip.write_gif(output)

# Perform inference on GIFs
files = [file for file in os.listdir(save_dir) if file.endswith('.gif')]
for file_name in files:
    file_name = save_dir + '/' + file_name
    video = load_gif(file_name, image_size=(244, 244))
    media.show_video(video.numpy(), fps=23)
    outputs = predict_top_k(model, video)
    for label, prob in outputs:
        print(label, prob)