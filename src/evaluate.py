import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_utils import FrameGenerator
from model_utils import load_pretrained_movinet

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

NUM_CLASSES = 25
classes_1 = ['Archery', 'BabyCrawling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'CliffDiving', 'CuttingInKitchen', 'Diving', 'Fencing', 'Hammering', 'HammerThrow', 'HighJump', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'LongJump', 'Mixing', 'MoppingFloor', 'PoleVault', 'Punch', 'SalsaSpin', 'Shotput', 'SumoWrestling', 'TaiChi', 'ThrowDiscus', 'TrampolineJumping']

model = tf.keras.models.load_model('saved_model/my_model')
test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], num_frames), output_signature=output_signature).batch(batch_size)

results = model.evaluate(test_ds, return_dict=True)
plot_history(results)

actual, predicted = get_actual_predicted_labels(model, test_ds)
plot_confusion_matrix(actual, predicted, classes_1)