"""
data_provider for hmdb51
"""
import os
import random
import sys
from Queue import Queue
from threading import Thread
import numpy as np
import cv2


class Data(object):
    """
    return Data with num_examples/next_batch/
    """
    def __init__(self, name, path, video_list, normalization, sequence_length,
                 crop_size, num_classes, queue_size):
        """
        Args:
        name: str, name of the data (train, test or validation)
        paths: list, list of string that have the video path and label information
        sequence_length: video clip length
        crop_size: `tuple`, image resize size (width, height)
        normalization: `str` or None
            None: no any normalization
            divide_255: divide all pixels by 255
            divide_256: divide all pixels by 256
        num_classes: `integer`, number of classes that the dataset has
        queue_size: `integer`, data queue size
        """
        self.name             = name
        self.path             = path
        self.video_list       = video_list
        self.normalization    = normalization
        self.sequence_length  = sequence_length
        self.crop_size        = crop_size
        self.num_classes      = num_classes
        self.queue            = DataQueue(name, queue_size)
        self.examples         = None
        self._start_data_thread()

    def get_frames_data(self, filename, sequence_length=16):
        ''' Given a directory containing extracted frames, return a video clip of
        (sequence_length) consecutive frames as a list of np arrays

        Args
        sequence_length: sequence_length of the video clip

        Returns
        video: numpy, video clip with shape
            [sequence_length, height, width, channels]
        '''
        video = []
        s_index = 0
        for _, _, files in os.walk(filename):
            filenames = [fi for fi in files]#if fi.endswith((".png", ".jpg", "jpeg"))]
            if len(filenames) < sequence_length:
                return None
            suffix = filenames[0].split('.', 1)[1]
            filenames_int = [i.split('.', 1)[0] for i in filenames]
            filenames_int = sorted(filenames_int)
            s_index = random.randint(0, len(filenames) - sequence_length)
            for i in range(s_index, s_index + sequence_length):
                image_name = str(filename) + '/' + str(filenames_int[i]) + '.' + suffix
               # print "image_name",image_name
                img = cv2.imread(image_name)
                img = cv2.resize(img, self.crop_size)
                if self.normalization:
                    img_data = self.normalize_image(img, self.normalization)
                video.append(img_data)
            return video

    def extract_video_data(self):
        ''' Single tread to extract video and label information from the dataset
        '''
        # Generate one randome index and
        while True:
            index = random.randint(0, len(self.video_list)-1)
            video_path, label = self.video_list[index].strip('\n').split()
            frame_path = os.path.join(self.path, 'hmdb51_frames/', video_path)
            dynmaic_path =os.path.join(self.path, 'hmdb51_dynamic/', video_path)
            # frame_path = os.path.join(self.path, 'hmdb10_frames/', video_path)
            # dynmaic_path =os.path.join(self.path, 'hmdb10_dynamic/', video_path)
            # print "frame_path",frame_path
            # print "dynmaic_path",dynmaic_path
            dynamic = self.get_frames_data(dynmaic_path, self.sequence_length)
            frames = self.get_frames_data(frame_path, self.sequence_length)
            if dynamic is not None and len(dynamic) == self.sequence_length and \
            frames is not None and len(frames) == self.sequence_length:
                # Put the video into the queue
                dynamic = np.array(dynamic)
                frames = np.array(frames)
                label = np.array(int(label))
                self.queue.put((dynamic, frames, label))

    def _start_data_thread(self):
        print("Start thread: %s data preparation ..." % self.name)
        self.worker = Thread(target=self.extract_video_data)
        self.worker.setDaemon(True)
        self.worker.start()

    @property
    def num_examples(self):
        '''
        return the number of examples to train/test
        '''
        if not self.examples:
        # calculate the number of examples
            total = 0
            for line in self.video_list:
                video_path, _ = line.strip('\n').split()
                #frame_path = os.path.join(self.path, 'hmdb10_frames/', video_path)
                frame_path = os.path.join(self.path, 'hmdb51_frames/', video_path)
                for _, _, files in os.walk(frame_path):
                    total += len(files)
            self.examples = total / self.sequence_length
        return self.examples*2

    def next_batch(self, batch_size):
        ''' Get the next batches of the dataset
        Args
        batch_size: video batch size
        Returns
        videos: numpy, shape
            [batch_size, sequence_length, height, width, channels]
        labels: numpy
            [batch_size, num_classes]
        '''
        dynamic, frames, labels = self.queue.get(batch_size)
        dynamic = np.array(dynamic)
        frames = np.array(frames)
        labels = np.array(labels)
        labels = self.labels_to_one_hot(labels, self.num_classes)
        return dynamic, frames, labels

    def normalize_image(self, img, normalization):
        """normalize image by 3 methods"""
        if normalization == 'std':
            if np.mean(img) == 0:
                print("it is ZERO!!!")
            img = (img - np.mean(img))/np.std(img)
        elif normalization == 'divide_256':
            img = img/256
        elif normalization == 'divide_255':
            img = img/255
        else:
            raise Exception("please set the norm method")
        return img

    def labels_to_one_hot(self, labels, num_classes):
        '''
        labels to one hot
        '''
        new_labels = np.zeros((labels.shape[0], num_classes))
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels

    def labels_for_one_hot(self, labels):
        '''
        labels for one hot
        '''
        return np.argmax(labels, axis=1)

class DataQueue(object):
    def __init__(self, name, maximum_item, block=True):
        """
        Args
        name: str, data type name (train, validation or test)
        maximum_item: integer, maximum item that this queue can store
        block: boolean, block the put or get information if the queue is
            full or empty
        """
        self._name         = name
        self.block        = block
        self.maximum_item = maximum_item
        self._queue        = Queue(maximum_item)

    @property
    def queue(self):
        return self._queue
    @property
    def name(self):
        return self._name

    def put(self, data):
        self._queue.put(data, self.block)

    def get(self, batch_size):
        '''
        Args:
        batch_size: integer, the number of the item you want to get from the queue
        Returns:
        videos: list, list of numpy data with shape
            [sequence_length, height, width, channels]
        labels: list, list of integer number
        '''
        dynamic = []
        frames = []
        labels = []
        for i in range(batch_size):
            dynamic_video, frames_video, label = self._queue.get(self.block)
            dynamic.append(dynamic_video)
            frames.append(frames_video)
            labels.append(label)
        return dynamic, frames, labels


class DataProvider(object):
    '''
    data_provider with train/valid/test
    return data_shape with data_provider.data_shape()
    return number of classes with data_provider.n_classes()
    '''
    def __init__(self, path, num_classes, validation_set=None, test=False,
                 validation_split=None, normalization=None, crop_size=(64, 64),
                 sequence_length=16, train_queue=None, valid_queue=None,
                 test_queue=None, train=False, queue_size=300, **kwargs):
        """
        Args:
        num_classes: the number of the classes
        validation_set: `bool`.
        validation_split: `int` or None
            float: chunk of `train set` will be marked as `validation set`.
            None: if 'validation set' == True, `validation set` will be
                copy of `test set`
        normalization: `str` or None
            None: no any normalization
            divide_255: divide all pixels by 255
            divide_256: divide all pixels by 256
        sequence_length: `integer`, video clip length
        crop_size: `tuple`, the size that you want to reshape the images, (width, height)
        train: `boolean`, whether we need the training queue or not
        test: `test`, whether we need the testing queue or not
        queue_size: `integer`, data queue size , default is 300
        """
        self._path            = path
        self._num_classes     = num_classes
        self._sequence_length = sequence_length
        self._crop_size       = crop_size

        # train_videos_labels   = self.get_videos_labels_lines(
        # os.path.join(self._path, 'train.list'))
        # test_videos_labels    = self.get_videos_labels_lines(
        # os.path.join(self._path, 'test.list'))

        train_videos_labels = self.get_path_and_label(os.path.join(self._path, 'train.list'))
        test_videos_labels = self.get_path_and_label(os.path.join(self._path, 'test.list'))

        if validation_set and validation_split:
            random.shuffle(train_videos_labels)
            valid_videos_labels = train_videos_labels[:validation_split]
            train_videos_labels = train_videos_labels[validation_split:]
            self.validation = Data('validation', self._path, valid_videos_labels,
                                    normalization, sequence_length,
                                    crop_size, num_classes, queue_size)
        if train:
            self.train = Data('train', self._path, train_videos_labels,
                                normalization, sequence_length,
                                crop_size, num_classes, queue_size)
        if test:
            self.test = Data('test', self._path, test_videos_labels,
                            normalization, sequence_length,
                            crop_size, num_classes, queue_size)
        if validation_set and not validation_split:
            self.validation = Data('validation', self._path, test_videos_labels,
                                    normalization, sequence_length,
                                    crop_size, num_classes, queue_size)

    def get_path_and_label(self, path):
        lines = open(path, 'r')
        lines = list(lines)
        return lines

    def get_videos_labels_lines(self, path):
        # Open the file according to the filename
        lines = open(path, 'r')
        lines = list(lines)
        new_lines = [os.path.join(self._path, line) for line in lines]
        return new_lines

    @property
    def data_shape(self):
        return (self._sequence_length, self._crop_size[1], self._crop_size[0], 3)

    @property
    def n_classes(self):
        return self._num_classes
