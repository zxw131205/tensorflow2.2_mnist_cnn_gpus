import numpy as np
import tensorflow as tf
import input_data

# 生成整数的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# MNIST数据集
mnist = input_data.read_data_sets('L:/data/mnist', dtype=tf.uint8, one_hot=False)
train_images = mnist.train.images
train_labels = mnist.train.labels
train_num_examples = mnist.train.num_examples

test_images = mnist.test.images
test_labels = mnist.test.labels
test_num_examples = mnist.test.num_examples

# 存储TFRecord文件的地址
filename1 = 'L:/record/train.tfrecords'
# 创建一个writer来写TFRecord文件
writer1 = tf.io.TFRecordWriter(filename1)

# 将每张图片都转为一个Example
for i1 in range(train_num_examples):
    image_raw = train_images[i1].tostring()  # 将图像转为字符串
    #label_raw=train_labels[i1].tostring()
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image': _bytes_feature(image_raw),
            'label': _int64_feature(train_labels[i1])
        }))
    writer1.write(example.SerializeToString())  # 将Example写入TFRecord文件

print('train data processing success')
writer1.close()

# 存储TFRecord文件的地址
filename2 = 'L:/record/test.tfrecords'
# 创建一个writer来写TFRecord文件
writer2 = tf.io.TFRecordWriter(filename2)
# 将每张图片都转为一个Example
for i2 in range(test_num_examples):
    image_raw1 = test_images[i2].tostring()  # 将图像转为字符串
    #label_raw1 = test_labels[i2].tostring()
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image': _bytes_feature(image_raw1),
            'label': _int64_feature(test_labels[i2])
        }))
    writer2.write(example.SerializeToString())  # 将Example写入TFRecord文件

print('test data processing success')
writer2.close()

import json
json_dict={"train_num_examples":train_num_examples,"test_num_examples":test_num_examples}

filename = 'L:/record/number.json'
with open(filename,'w') as f_obj:
    json.dump(json_dict,f_obj)
f_obj.close()



example = tf.train.Example(features=tf.train.Features(feature={
            'train': tf.train.Feature(mnist.train),
            'test': tf.train.Feature(mnist.test),
            'length': (_int64_feature(train_num_examples),_int64_feature(test_num_examples))
        }
        ))


writer.write(example.SerializeToString())

features = FeaturesDict({
    'image': Image(shape=(28, 28, 1), dtype=tf.uint8),
    'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=10),
}),
total_num_examples = 70000,
splits = {
    'test': 10000,
    'train': 60000,
}

for i in range(train_num_examples):
    # Read raw image
    img = tf.compat.v1.read_file(filename).numpy()
    # Parse its label from the filename
    label = int(filename.split('_')[-1].split('.')[0])
    # Create an example (image, label)
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'image': _bytes_feature(img)}))
    # Write serialized example to TFRecords
    writer.write(example.SerializeToString())