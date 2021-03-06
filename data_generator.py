# Lint as: python2, python3
# Copyright 2020 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Wrapper for providing semantic segmentaion data.

The SegmentationDataset class provides both images and annotations (semantic
segmentation and/or instance segmentation) for TensorFlow. Currently, we
support the following datasets:

1. PASCAL VOC 2012 (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

PASCAL VOC 2012 semantic segmentation dataset annotates 20 foreground objects
(e.g., bike, person, and so on) and leaves all the other semantic classes as
one background class. The dataset contains 1464, 1449, and 1456 annotated
images for the training, validation and test respectively.

2. Cityscapes dataset (https://www.cityscapes-dataset.com)

The Cityscapes dataset contains 19 semantic labels (such as road, person, car,
and so on) for urban street scenes.

3. ADE20K dataset (http://groups.csail.mit.edu/vision/datasets/ADE20K)

The ADE20K dataset contains 150 semantic labels both urban street scenes and
indoor scenes.

References:
  M. Everingham, S. M. A. Eslami, L. V. Gool, C. K. I. Williams, J. Winn,
  and A. Zisserman, The pascal visual object classes challenge a retrospective.
  IJCV, 2014.

  M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson,
  U. Franke, S. Roth, and B. Schiele, "The cityscapes dataset for semantic urban
  scene understanding," In Proc. of CVPR, 2016.

  B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, A. Torralba, "Scene Parsing
  through ADE20K dataset", In Proc. of CVPR, 2017.
"""

import collections
import os
import tensorflow.compat.v1 as tf
import tensorflow as tf_sess
from third_party.deeplab import common
from core import input_preprocess

import copy_paste
import random
from tensorflow import contrib
autograph = contrib.autograph

import sys
sys.setrecursionlimit(20000) #设置递归深度
#import numpy as np
#np.set_printoptions(threshold=np.inf)

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',  # Number of semantic classes, including the
                        # background class (if exists). For example, there
                        # are 20 foreground classes + 1 background class in
                        # the PASCAL VOC 2012 dataset. Thus, we set
                        # num_classes=21.
        'ignore_label',  # Ignore label value.
    ])

_CITYSCAPES_INFORMATION = DatasetDescriptor(
    splits_to_sizes={'train_fine': 2975,
                     'train_coarse': 22973,
                     'trainval_fine': 3475,
                     'trainval_coarse': 23473,
                     'val_fine': 500,
                     'test_fine': 1525},
    num_classes=19,
    ignore_label=255,
)

# To generate the tfrecord, please refer to
# https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/download_and_convert_voc2012.sh
_PASCAL_VOC_SEG_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 1464,
        'train_aug': 10582,
        'trainval': 2913,
        'val': 1449,
        # Splits for semi-supervised
        '4_clean': 366,
        '8_clean': 183,
        # Balanced 1/16 split
        # Sample with rejection, suffix represents the sample index
        # e.g., 16_clean_3 represents the 3rd random shuffle to sample 1/16
        # split, given a fixed random seed 8888
        '16_clean_3': 92,
        '16_clean_14': 92,
        '16_clean_20': 92,
        # More images
        'coco2voc_labeled': 91937,
        'coco2voc_unlabeled': 215340,
    },
    num_classes=21,
    ignore_label=255,
)

_ADE20K_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 20210,  # num of samples in images/training
        'val': 2000,  # num of samples in images/validation
    },
    num_classes=151,
    ignore_label=0,
)

_COCO_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 118287,
        'val': 5000,
        # Splits for semi-supervised
        # e.g., 32_all represents 1/32 split
        '32_all': 3697,
        '64_all': 1849,
        '128_all': 925,
        '256_all': 463,
        '512_all': 232,
        'unlabeled': 123403,
    },
    num_classes=81,
    ignore_label=255,
)

_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'coco': _COCO_INFORMATION,
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'


def get_cityscapes_dataset_name():
  return 'cityscapes'


class Dataset(object):
  """Represents input dataset for deeplab model."""

  def __init__(self,
               dataset_name,
               split_name,
               dataset_dir,
               batch_size,
               crop_size,
               min_resize_value=None,
               max_resize_value=None,
               resize_factor=None,
               min_scale_factor=1.,
               max_scale_factor=1.,
               scale_factor_step_size=0,
               model_variant=None,
               num_readers=1,
               is_training=False,
               should_shuffle=False,
               should_repeat=False,
               with_cls=False,
               cls_only=False,
               copy_paste=False,
               lsj=True,
               strong_weak=False,
               output_valid=False,
               output_original=True):
    """Initializes the dataset.

    Args:
      dataset_name: Dataset name.
      split_name: A train/val Split name.
      dataset_dir: The directory of the dataset sources.
      batch_size: Batch size.
      crop_size: The size used to crop the image and label.
      min_resize_value: Desired size of the smaller image side.
      max_resize_value: Maximum allowed size of the larger image side.
      resize_factor: Resized dimensions are multiple of factor plus one.
      min_scale_factor: Minimum scale factor value.
      max_scale_factor: Maximum scale factor value.
      scale_factor_step_size: The step size from min scale factor to max scale
        factor. The input is randomly scaled based on the value of
        (min_scale_factor, max_scale_factor, scale_factor_step_size).
      model_variant: Model variant (string) for choosing how to mean-subtract
        the images. See feature_extractor.network_map for supported model
        variants.
      num_readers: Number of readers for data provider.
      is_training: Boolean, if dataset is for training or not.
      should_shuffle: Boolean, if should shuffle the input data.
      should_repeat: Boolean, if should repeat the input data.
      with_cls: Boolean, True if we use it for CAM (classification) training
      cls_only: Boolean, True if we only want to keep image-level label
      strong_weak: Output both weak augmented and strong augmented images or not
      output_valid: Output the valid mask to exclude padding region or not

    Raises:
      ValueError: Dataset name and split name are not supported.
    """
    if dataset_name not in _DATASETS_INFORMATION:
      raise ValueError('The specified dataset is not supported yet.')
    self.dataset_name = dataset_name

    splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

    if split_name not in splits_to_sizes:
      raise ValueError('data split name %s not recognized' % split_name)

    if model_variant is None:
      tf.logging.warning('Please specify a model_variant. See '
                         'feature_extractor.network_map for supported model '
                         'variants.')

    self.split_name = split_name
    self.dataset_dir = dataset_dir
    self.batch_size = batch_size
    self.crop_size = crop_size
    self.min_resize_value = min_resize_value
    self.max_resize_value = max_resize_value
    self.resize_factor = resize_factor
    self.min_scale_factor = min_scale_factor
    self.max_scale_factor = max_scale_factor
    self.scale_factor_step_size = scale_factor_step_size
    self.model_variant = model_variant
    self.num_readers = num_readers
    self.is_training = is_training
    self.should_shuffle = should_shuffle
    self.should_repeat = should_repeat
    self.cls_only = cls_only
    if cls_only:
      self.with_cls = True
    else:
      self.with_cls = with_cls
    self.copy_paste = copy_paste
    self.lsj = lsj
    self.strong_weak = strong_weak
    self.output_valid = output_valid
    self.output_original = output_original

    self.num_of_classes = _DATASETS_INFORMATION[self.dataset_name].num_classes
    self.ignore_label = _DATASETS_INFORMATION[self.dataset_name].ignore_label

  # return zhe information of one sample in the dataset
  # including parsed image, label, height, width and image name
  def _parse_function(self, example_proto):
    """Function to parse the example proto.

    Args:
      example_proto: Proto in the format of tf.Example.

    Returns:
      A dictionary with parsed image, label, height, width and image name.

    Raises:
      ValueError: Label is of wrong shape.
    """

    # Currently only supports jpeg and png.
    # Need to use this logic because the shape is not known for
    # tf.image.decode_image and we rely on this info to
    # extend label if necessary.
    def _decode_image(content, channels):
      return tf.cond(
          tf.image.is_jpeg(content),
          lambda: tf.image.decode_jpeg(content, channels),
          lambda: tf.image.decode_png(content, channels))

    features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/segmentation/class/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    parsed_features = tf.parse_single_example(example_proto, features)

    image = _decode_image(parsed_features['image/encoded'], channels=3)

    label = None
    if self.split_name != common.TEST_SET:
      label = _decode_image(
          parsed_features['image/segmentation/class/encoded'], channels=1)

    image_name = parsed_features['image/filename']
    if image_name is None:
      image_name = tf.constant('')

    sample = {
        common.IMAGE: image,
        common.IMAGE_NAME: image_name,
        common.HEIGHT: parsed_features['image/height'],
        common.WIDTH: parsed_features['image/width'],
    }

    if label is not None:
      if label.get_shape().ndims == 2:
        label = tf.expand_dims(label, 2)
        #expand [height, width] to [height, width, 1]
      elif label.get_shape().ndims == 3 and label.shape.dims[2] == 1:
        pass
      else:
        raise ValueError('Input label shape must be [height, width], or '
                         '[height, width, 1].')

      label.set_shape([None, None, 1])

      sample[common.LABELS_CLASS] = label

    return sample

  def _preprocess_image(self, sample):
    """Preprocesses the image and label.

    Args:
      sample: A sample containing image and label.

    Returns:
      sample: Sample with preprocessed image and label.

    Raises:
      ValueError: Ground truth label not provided during training.
    """
    image = sample[common.IMAGE]
    label = sample[common.LABELS_CLASS]

    if not self.strong_weak:
      if not self.output_valid:
        original_image, image, label = input_preprocess.preprocess_image_and_label(
            image=image,
            label=label,
            crop_height=self.crop_size[0],
            crop_width=self.crop_size[1],
            min_resize_value=self.min_resize_value,
            max_resize_value=self.max_resize_value,
            resize_factor=self.resize_factor,
            min_scale_factor=self.min_scale_factor,
            max_scale_factor=self.max_scale_factor,
            scale_factor_step_size=self.scale_factor_step_size,
            ignore_label=self.ignore_label,
            is_training=self.is_training,
            model_variant=self.model_variant)
      else:
        original_image, image, label, valid = input_preprocess.preprocess_image_and_label(
            image=image,
            label=label,
            crop_height=self.crop_size[0],
            crop_width=self.crop_size[1],
            min_resize_value=self.min_resize_value,
            max_resize_value=self.max_resize_value,
            resize_factor=self.resize_factor,
            min_scale_factor=self.min_scale_factor,
            max_scale_factor=self.max_scale_factor,
            scale_factor_step_size=self.scale_factor_step_size,
            ignore_label=self.ignore_label,
            is_training=self.is_training,
            model_variant=self.model_variant,
            output_valid=self.output_valid)
        sample['valid'] = valid
    else:
      original_image, image, label, strong, valid = input_preprocess.preprocess_image_and_label(
          image=image,
          label=label,
          crop_height=self.crop_size[0],
          crop_width=self.crop_size[1],
          min_resize_value=self.min_resize_value,
          max_resize_value=self.max_resize_value,
          resize_factor=self.resize_factor,
          min_scale_factor=self.min_scale_factor,
          max_scale_factor=self.max_scale_factor,
          scale_factor_step_size=self.scale_factor_step_size,
          ignore_label=self.ignore_label,
          is_training=self.is_training,
          model_variant=self.model_variant,
          strong_weak=self.strong_weak)
      sample['strong'] = strong
      sample['valid'] = valid

    sample[common.IMAGE] = image

    if not self.is_training and self.output_original:
      # Original image is only used during visualization.
      sample[common.ORIGINAL_IMAGE] = original_image

    if label is not None:
      sample[common.LABEL] = label

    # Remove common.LABEL_CLASS key in the sample since it is only used to
    # derive label and not used in training and evaluation.
    sample.pop(common.LABELS_CLASS, None)

    # Convert segmentation map to multi-class label
    if self.with_cls and label is not None:
      base = tf.linalg.LinearOperatorIdentity(
          num_rows=self.num_of_classes - 1, dtype=tf.float32)
      base = base.to_dense()
      zero_filler = tf.zeros([1, self.num_of_classes-1], tf.float32)
      base = tf.concat([zero_filler, base], axis=0)

      cls = tf.unique(tf.reshape(label, shape=[-1]))[0]
      select = tf.less(cls, self.ignore_label)
      cls = tf.boolean_mask(cls, select)
      cls_label = tf.reduce_sum(tf.gather(base, cls, axis=0), axis=0)
      sample['cls_label'] = tf.stop_gradient(cls_label)

    if self.cls_only:
      del sample[common.LABEL]

    return sample

  def get_one_shot_iterator(self):
    """Gets an iterator that iterates across the dataset once.

    Returns:
      An iterator of type tf.data.Iterator.
    """
    files = self._get_all_files()

    if not self.copy_paste:
      dataset = (
          tf.data.TFRecordDataset(files, num_parallel_reads=self.num_readers)
          .map(self._parse_function, num_parallel_calls=self.num_readers)
          .map(self._preprocess_image, num_parallel_calls=self.num_readers))
    # map(map_func, num_parallel_calls=None, deterministic=None)
    # This transformation applies map_func to each element of this dataset,
    # and returns a new dataset containing the transformed elements,
    # in the same order as they appeared in the input.
    else:
      dataset = (
          tf.data.TFRecordDataset(files, num_parallel_reads=self.num_readers)
          .map(self._parse_function, num_parallel_calls=self.num_readers))
          #.map(self._preprocess_image, num_parallel_calls=self.num_readers))

      num = _DATASETS_INFORMATION[self.dataset_name].splits_to_sizes[self.split_name]
      dataset_temp = dataset.shuffle(buffer_size=100)
      #这里shuffle可以做一个超参数看影响！！！
      #list = []
      #list_dataset = []
      # dataset = dataset.shuffle(buffer_size=100)
      dataset_fin = dataset.skip(num+10)

      iterator = dataset.make_one_shot_iterator()
      next_element = iterator.get_next()

      iterator_temp = dataset_temp.make_one_shot_iterator()
      next_element_temp = iterator_temp.get_next()

#替代方案，先对数据集作用，再生成tfrecord读取
      cp = copy_paste.copy_paste
      cp = autograph.to_graph(cp)
      i=0

      with tf.Session() as sess:
          try:
            while True:
              print(i)
              i = i+1
              sample = sess.run(next_element)
              sample_src = sess.run(next_element_temp)
              image_main = sample[common.IMAGE]
              mask_main = sample[common.LABELS_CLASS]
              mask_src = sample_src[common.LABELS_CLASS]
              image_src = sample_src[common.IMAGE]

              label, image = cp(
                  mask_src, image_src, mask_main, image_main, lsj=self.lsj)

              #image = tf.cast(image, tf.uint8)
              #label = tf.cast(label, tf.uint8)
              sample[common.IMAGE] = image
              sample[common.LABELS_CLASS] = label

              dataset_t = tf.data.Dataset.from_tensors(sample)
              dataset_fin = dataset_fin.concatenate(dataset_t)

          except tf_sess.errors.OutOfRangeError:
              print("end!end!")

      #for sample in dataset:

      dataset =dataset_fin.map(self._preprocess_image, num_parallel_calls=self.num_readers)
      #print("dataset_final")
    if self.should_shuffle:
      dataset = dataset.shuffle(buffer_size=100)

    if self.should_repeat:
      dataset = dataset.repeat()  # Repeat forever for training.
    else:
      dataset = dataset.repeat(1)

    if not self.output_original and not self.is_training:
      dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(self.batch_size)
    else:
      dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)
    #print("final")
    return dataset.make_one_shot_iterator()

  def _get_all_files(self):
    """Gets all the files to read data from.

    Returns:
      A list of input files.
    """
    file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(self.dataset_dir,
                                file_pattern % self.split_name)
    return tf.gfile.Glob(file_pattern)
    #return all the files which match the form of 'split_name-*' as a list

