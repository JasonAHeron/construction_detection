__author__ = 'HANEL'

'''
  Simple library to read all PNG and JPG/JPEG images in a directory
  with TensorFlow buil-in functions to boost speed.
  Supported formats by TensorFlow are: PNG, JPG/JPEG
  Hamed MP
  Github: @hamedmp
  Twitter: @hamedpc2002
'''

import os
import glob
import tensorflow as tf
from PIL import Image
import numpy as np


def read_images(path, is_directory=True):

  images = []
  png_files = []
  jpeg_files = []

  reader = tf.WholeFileReader()

  png_files_path = glob.glob(os.path.join(path, '*.[pP][nN][gG]'))
  jpeg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][eE][gG]'))
  jpg_files_path = glob.glob(os.path.join(path, '*.[jJ][pP][gG]'))

  if is_directory:
    for filename in png_files_path:
      png_files.append(filename)
    for filename in jpeg_files_path:
      jpeg_files.append(filename)
    for filename in jpg_files_path:
      jpeg_files.append(filename)
  else:
    raise ValueError('Currently only batch read from directory supported')


  # Decode if there is a PNG file:
  if len(png_files) > 0:
    png_file_queue = tf.train.string_input_producer(png_files)
    pkey, pvalue = reader.read(png_file_queue)
    p_img = tf.image.decode_png(pvalue)

  if len(jpeg_files) > 0:
    jpeg_file_queue = tf.train.string_input_producer(jpeg_files)
    jkey, jvalue = reader.read(jpeg_file_queue)
    j_img = tf.image.decode_jpeg(jvalue)


  with tf.Session() as sess:

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    if len(png_files) > 0:
      for i in range(len(png_files)):
        png = p_img.eval()
        images.append(png)


    if len(jpeg_files) > 0:
      for i in range(len(jpeg_files)):
        jpeg = j_img.eval().flatten()
        images.append(jpeg)

    coord.request_stop()
    coord.join(threads)

  return np.asarray(images)
