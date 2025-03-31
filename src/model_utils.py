import tensorflow as tf
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

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