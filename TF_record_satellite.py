import tensorflow as tf 
from io import BytesIO
import numpy as np
import argparse

#Path for the data to record
file_name = ['train_data.tfrecord']
img_path = './meteo_data'

list_input = np.array([])
count = 0

#List of bytes of separated images for each day and each hour
for pca_img in os.lisdir(os.path.join(img_path, 'input'):
    img = np.load(os.path.join(img_path, 'input', pca_img))
    for day in img.shape[0]:
        for hour in img.shape[1]:
            list_input.append(img[day][hour].flatten())
            count+=1

#Conversion to bytes and definition of features
data_LR = list_input.tobytes()
index = count
c = img.shape[-1]
h_LR = img.shape[2]
w_LR = img.shape[3]

#Definition of feature types
feature = {
      'c': _int64_feature(c),
      'h_LR': _int64_feature(h_LR),
      'w_LR ': _int64_feature(w_LR ),
      'index': _int64_feature(index),
      'data_LR': _bytes_feature(data_LR),
  }

#Writing of tf_record
writer = tf.io.TFRecordWriter(file_name)
example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
example = example_proto.SerializeToString()
writer.write(example)


