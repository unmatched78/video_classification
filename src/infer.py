import os
import random
import tensorflow as tf
from moviepy.editor import VideoFileClip
import mediapy as media
from data_utils import load_gif
from model_utils import load_pretrained_movinet

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

NUM_CLASSES = 25
classes_1 = ['Archery', 'BabyCrawling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'CliffDiving', 'CuttingInKitchen', 'Diving', 'Fencing', 'Hammering', 'HammerThrow', 'HighJump', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'LongJump', 'Mixing', 'MoppingFloor', 'PoleVault', 'Punch', 'SalsaSpin', 'Shotput', 'SumoWrestling', 'TaiChi', 'ThrowDiscus', 'TrampolineJumping']

model = tf.keras.models.load_model('saved_model/my_model')
save_dir = '/content/video_classification/sample_gifs'
files = [file for file in os.listdir(save_dir) if file.endswith('.gif')]

for file_name in files:
    file_name = save_dir + '/' + file_name
    video = load_gif(file_name, image_size=(244, 244))
    media.show_video(video.numpy(), fps=23)
    outputs = predict_top_k(model, video)
    for label, prob in outputs:
        print(label, prob)