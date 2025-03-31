import pathlib
import tensorflow as tf
from data_utils import FrameGenerator, download_ucf_101_subset
from model_utils import build_classifier, load_pretrained_movinet

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

model, backbone = load_pretrained_movinet()
model = build_classifier(batch_size, num_frames, 224, backbone, NUM_CLASSES)

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

num_epochs = 2
results = model.fit(train_ds, validation_data=val_ds, epochs=num_epochs, validation_freq=1, verbose=1)
model.save('saved_model/my_model')