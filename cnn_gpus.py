import tensorflow as tf
import numpy as np
import json

import os
import json
print(tf.__version__)


def parser(record):
    '''Function to parse a TFRecords example'''

    # Define here the features you would like to parse
    features = {'image': tf.io.FixedLenFeature((), tf.string),
                'label': tf.io.FixedLenFeature((), tf.int64)}

    # Parse example
    parsed = tf.io.parse_single_example(record, features)

    # Decode image
    img = tf.io.decode_raw(parsed['image'], tf.uint8)
    img = tf.reshape(img, [28, 28,1])
    img = img / 255

    #label=tf.io.decode_raw(parsed['label'], tf.uint8)
    return img, parsed['label']

filename1 = ['L:/record/train.tfrecords']
mnist_train = tf.data.TFRecordDataset(filename1)


filename2 = ['L:/record/test.tfrecords']
mnist_test = tf.data.TFRecordDataset(filename2)
#mnist_test =parser(filename2)


# You can also do info.splits.total_num_examples to get the total number of examples in the dataset.
filename = 'L:/record/number.json'
with open(filename,'r') as f_obj:
    num=json.load(f_obj)

num_train_examples = num['train_num_examples']
num_test_examples = num['test_num_examples']

strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

train_dataset = mnist_train.map(parser).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(parser).batch(BATCH_SIZE)


with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

# Define the checkpoint directory to store the checkpoints
path='L:/model'
checkpoint_dir = os.path.join(path,'/training_checkpoints')
logdir=os.path.join(path,'/logs')
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=logdir),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

model.fit(train_dataset, epochs=12, callbacks=callbacks)

# check the checkpoint directory
#cmd
#if exist checkpoint_dir" echo checkpoint_dir
#linux
#!ls {checkpoint_dir}

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(eval_dataset)

print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

#tensorboard --logdir=path/to/log-directory

#ls -sh ./logs


save_path = os.path.join(path,'saved_model/')

model.save(save_path, save_format='tf')


unreplicated_model = tf.keras.models.load_model(save_path)

unreplicated_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy'])

eval_loss, eval_acc = unreplicated_model.evaluate(eval_dataset)

print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))


with strategy.scope():
  replicated_model = tf.keras.models.load_model(path)
  replicated_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])

  eval_loss, eval_acc = replicated_model.evaluate(eval_dataset)
  print ('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

