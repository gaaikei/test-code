import os
import random
import tempfile
from Queue import Queue
from threading import Thread
import cv2
import numpy as np



class Data(object):
    """Data provide data for model,
    which has two type of images(rgb and dynamic)."""
    def __init__(self, name, path, normalization,
    frames_per_clip, crop_size, num_classes, queue_size):
        self.name = name
        self.path = path
        self.frames_per_clip = frames_per_clip
        self.crop_size = crop_size
        self.normalization = normalization
        self.num_classes = num_classes
        self.queue = DataQueue(name, queue_size)
        self._start_data_thread()

    def get_video_data(self):
        

    def get_frames(self, filename, frames_per_clip=16):
        video=[]

    def _start_data_thread(self):
        print('start thread : %s data preparation ... '% self.name)
        self.worker = Thread(target=self.get_video_data)
        self.worker = setDaemon(True)
        self.worker.start()