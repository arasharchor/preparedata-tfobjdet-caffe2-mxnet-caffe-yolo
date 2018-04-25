# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw dota dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_dota_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --year=VOC2012 \
        --set=train \
        --output_path=/home/user/dota.record
    python create_dota_tf_record.py \
            --data_dir=VOCdevkit \
            --year=VOC2012 \
            --set=train \
            --output_path=dota_train.record
    python create_dota_tf_record.py \
                --data_dir=VOCdevkit \
                --year=VOC2012 \
                --set=val \
                --output_path=dota_val.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image
import tensorflow as tf

import dataset_util
import label_map_util

import xml.etree.ElementTree as ET
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw dota VOC dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/dota_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']


def dict_to_tf_example(path,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):

  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding dota XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding dota dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      dota dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['imgfilename'] is not a valid JPEG
  """
  tree = ET.parse(path)
  objs = tree.findall('object')
  source = tree.find('source')
  imagesize = source.find('imagesize')
  height = int(imagesize.find('nrows').text)
  width = int(imagesize.find('ncols').text)
  num_objs = len(objs)
  filename = tree.find('filename').text
  print(filename)

  img_path = os.path.join('VOC2012', 'JPEGImages', filename)
  print(img_path)
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()
  #print(data['source']['imagesize']['ncols'])

  ## TODO changed here
  # Load object bounding boxes into a data frame.
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for ix, obj in enumerate(objs):
      x1 = []
      y1 = []
      x2 = []
      y2 = []
      xs = []
      ys = []
      difficult = bool(int(obj.find('difficult').text))
      print('difficult is {}'.format(int(obj.find('difficult').text))
      #if ignore_difficult_instances and difficult:
      #   continue
      difficult_obj.append(int(difficult))

      for pt in obj.findall('pt'):
          x = float(pt.find('x').text)
          y = float(pt.find('y').text)
          xs.extend([x])
          ys.extend([y])
      print('xs is {}'.format(xs))
      print('ys is {}'.format(ys))


      if sum([1 for x in xs if not 1 <= x <= width]) > 3:
          if not ((sum([1 for x in xs if not x < 0]) == 2) and (sum([1 for x in xs if not x > width]) == 2)):
             print(xs)
             print(ys)
             print(tree.find('imgfilename').text)
             print(obj.find('name').text)
             print(height)
             print(width)
             raise AssertionError('x points are out')


      if sum([1 for y in ys if not 1 <= y <= height]) > 3:
          if not ((sum([1 for y in ys if not y < 0]) == 2) and (sum([1 for y in ys if not y > height]) == 2)):
             print(xs)
             print(ys)
             print(tree.find('imgfilename').text)
             print(obj.find('name').text)
             print(height)
             print(width)
             raise AssertionError('y points are out')

      ############## HBB
      #xss = [min(xs), max(xs), max(xs), min(xs)]
      #yss = [min(ys), min(ys), max(ys), max(ys)]
      #x1 = float(xss[0])
      #y1 = float(yss[0])
      #x2 = float(xss[2])
      #y2 = float(yss[2])

      x1, y1, x2, y2 = [float(min(xs)), float(mix(ys)), float(max(xs)), float(max(xs))]

      if x1 < 1.0: x1 = 1.0
      if y1 < 1.0: y1 = 1.0
      if x2 > width: x2 = width
      if y2 > height: y2 = height
      # Make pixel indexes 0-based original faster rcnn
      #x1 -= 1
      #y1 -= 1
      #x2 -= 1
      #y2 -= 1
      print('read is {}, {}, {}, {}'.format(x1, y1, x2, y2))
      #float(obj['bndbox']['ymin'])
      #float(obj['bndbox']['xmax'])
      #float(obj['bndbox']['ymax'])

      print 'read coordinates are xmin {}, ymin {}, xmax{}, ymax{}'.format(x1,y1,x2,y2)

      # TODO check division by width and height should be there or no
      xmin.append(x1)# / width)
      ymin.append(y1)# / height)
      xmax.append(x2)# / width)
      ymax.append(y2)# / height)
      print('classes_text is {}'.format(obj.find('name').text.encode('utf8')))
      classes_text.append(obj.find('name').text.encode('utf8'))
      print('num for classes is {}'.format(label_map_dict[obj.find('name').text]))
      classes.append(label_map_dict[obj.find('name').text])
      truncated.append(0)#int(obj['truncated'])
      poses.append('top'.encode('utf8')) #obj['pose'].encode('utf8')


  print('classes is {} in tf.record {}'.format(obj.find('name').text, label_map_dict[obj.find('name').text]))
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/imgfilename': dataset_util.bytes_feature(
          filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          filename.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  if FLAGS.year not in YEARS:
    raise ValueError('year must be in : {}'.format(YEARS))

  data_dir = FLAGS.data_dir
  years = ['VOC2007', 'VOC2012']
  if FLAGS.year != 'merged':
    years = [FLAGS.year]

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  #print(label_map_dict)
  for year in years:
    logging.info('Reading from dota %s dataset.', year)
    examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                                 FLAGS.set + '.txt')
    annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)
    examples_list = dataset_util.read_examples_list(examples_path)
    print(annotations_dir)
    print(len(examples_list))
    counter = 0
    for idx, example in enumerate(examples_list):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples_list))
      path = os.path.join(annotations_dir, example + '.xml')
      print('path is {}'.format(path))
      counter+=1
      #print(counter)
      #with tf.gfile.GFile(path, 'r') as fid:
      #  xml_str = fid.read()
      #xml = etree.fromstring(xml_str)
      #data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

      tf_example = dict_to_tf_example(path, FLAGS.data_dir, label_map_dict,
                                      FLAGS.ignore_difficult_instances)
      writer.write(tf_example.SerializeToString())
  print(counter)
  writer.close()


if __name__ == '__main__':
  tf.app.run()
