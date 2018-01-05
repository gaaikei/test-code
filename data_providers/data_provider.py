import os
import random
import sys
from Queue import Queue
from threading import Thread
import numpy as np
import cv2

class Data(object):
    """docstring for Data"""
    def __init__(self, name, my_indexes, path, normalization, sequence_length, crop_size,
                 num_classes, queue_size):

        self.path = path
        self.name = name
        self.sequence_length = sequence_length
        self.crop_size = crop_size
        self.normalization = normalization
        self.num_classes = num_classes
        self.queue = DataQueue(name, queue_size)
        self.examples = None
        #self.num = 0
        self.my_indexes = my_indexes
        self._start_data_thread()


    def _start_data_thread(self):
        # print("start thread: %s data preparation ..." % self.name)
        self.worker = Thread(target=self.get_video_data)
        self.worker.setDaemon(True)
        self.worker.start()

    def get_video_data(self):
        my_num = 0
        ind_len = len(self.path)
        print "len of indexes,in get_video_data:",len(self.my_indexes)
        while True:
            #len(self.path)=1710(hmdb51)
            # index = random.randint(0, len(self.path)-1)
            # print "path[index]",self.path[index]
        # print "len(self.path)=",len(self.path)
            if my_num != ind_len-1:
                my_num += 1
            elif my_num >= ind_len-1:
                my_num = 0
            ind = self.my_indexes[my_num]
             #for index in range(len(self.path)-1):
            video_path, label = self.path[ind].strip('\n').split()
            # print "video_path",video_path
            video = self.get_frames(video_path, self.sequence_length)
            if video is not None and len(video) == self.sequence_length:
                video = np.array(video)
                label = np.array(int(label))
                self.queue.put((video, label))


    def get_video_lists(self, path, video_list, type_path):
        lines = video_list
        new_lines = [os.path.join(self.path, type_path, line) for line in lines]
        return new_lines

    def get_frames(self, filename, sequence_length=16):
        video = []
        start = 0
        # print "filename[1]",filename[1]
        for parent, dirnames, files in os.walk(filename):
            # print "filename:",filename
            filenames = [file for file in files ]#if file.endswith(".jpeg", ".jpg", ".png")
            if len(filenames) < sequence_length:
                # print "filenames<16"
                return  None
            suffix = filenames[0].split('.', 1)[1]
            filenames_int = [i.split('.', 1)[0] for i in filenames]
            filenames_int = sorted(filenames_int)
            start = random.randint(0, len(filenames) - sequence_length)
            for i in range(start, start+sequence_length):
                img_name = str(filename) + '/' + str(filenames_int[i]) + '.' + suffix
                img = cv2.imread(img_name)
                img = cv2.resize(img, self.crop_size)
                if self.normalization:
                    img = self.normalize_image(img, self.normalization)
                video.append(img)
            return video


    def normalize_image(self, img, normalization):
        """normalize image by 3 methods"""
        if normalization == 'std':
            img = (img - np.mean(img))/np.std(img)
        elif normalization == 'divide_256':
            img = img/256
        elif normalization == 'divide_255':
            img = img/255
        else:
            raise Exception("please set the norm method")
        return img

    @property
    def num_examples(self):
        if not self.examples:
            total = 0
            for line in self.path:
                #print "line:",line
                video_path, _ = line.strip('\n').split()
                for root, dirs, files in os.walk(video_path):
                    total += len(files)
            self.examples = total / self.sequence_length
        return self.examples * 2

    def next_batch(self, batch_size):
        videos, labels = self.queue.get(batch_size)
        videos = np.array(videos)
        labels = np.array(labels)
        labels = self.labels_to_one_hot(labels, self.num_classes)
        return videos, labels

    def labels_to_one_hot(self, labels, num_classes):
        # labels = labels
        new_labels = np.zeros((labels.shape[0], num_classes))
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels
    def labels_for_one_hot(self, labels):
        return np.argmax(labels, axis=1)




class DataQueue(object):
    """docstring for Data"""
    def __init__(self, name, max_items, block=True):
        # print "init DataQueue:"
        # print "name :",name
        # print "max_items:",max_items
        self._name = name
        self.block = block
        self.max_items = max_items
        self._queue = Queue(max_items)

    @property
    def queue(self):
        return self._queue
    @property
    def name(self):
        return self._name
    def put(self, data):
        self._queue.put(data, self.block)
    def get(self, batch_size):
        videos = []
        labels = []
        for i in range(batch_size):
            video, label = self._queue.get(self.block)
            videos.append(video)
            labels.append(label)
        return videos, labels




class DataProvider(object):
    """get three types of data"""
    def __init__(self, path, num_classes, validation_set=None, test=False,
                 validation_split=None, normalization=None, crop_size=(64, 64),
                 train_queue=None, valid_queue=None, test_queue=None, sequence_length=16,
                 train=False, queue_size=200, **kwargs):
        """
        path: path to data
        num_classes: number of classes
        validation_set : 'bool'
        validation_split : 'int' or None

        """
        self._path = path
        self._num_classes = num_classes
        self._crop_size = crop_size
        self._sequence_length = sequence_length

        print "in data_provider: self.path",self._path
        train_labels = self.get_path_and_label(os.path.join(self._path, 'train.list'))
        test_labels = self.get_path_and_label(os.path.join(self._path, 'test.list'))

        if validation_set and validation_split:
            #shuffle data
            # print "1"
            random.shuffle(train_labels)
            valid_labels = train_labels[:validation_split]
            train_labels = train_labels[validation_split:]
            self.validation = twoStreamData('validation',self._path, valid_labels, normalization,
                                           sequence_length, crop_size, num_classes, queue_size)
        if train:
            # print "2"
            self.train = twoStreamData('train', self._path,train_labels, normalization, sequence_length,
                                      crop_size, num_classes, queue_size)
        if test:
            # print "3"
            self.test = twoStreamData('test',self._path, test_labels, normalization, sequence_length,
                                     crop_size, num_classes, queue_size)

        if validation_set and not validation_split:
            # print "4"
            self.validation = twoStreamData('validation',self._path, test_labels,
                normalization, sequence_length, crop_size, num_classes, queue_size)


    def get_path_and_label(self, path):
        lines = open(path, 'r')
        lines = list(lines)
        return lines


    @property
    def data_shape(self):
        """return data's shape"""
        return (self._sequence_length, self._crop_size[1], self._crop_size[0], 3)

    @property
    def n_classes(self):
        """return number of classes"""
        return self._num_classes


class twoStreamData(object):
    """docstring for twoStreamData"""
    def __init__(self, name, path, video_list, normalization, sequence_length, crop_size,
                 num_classes, queue_size):
        self.path = path
        self.name = name
        self.video_list = video_list
        self.sequence_length = sequence_length
        self.crop_size = crop_size
        self.normalization = normalization
        self.num_classes = num_classes
        self.queue_size = queue_size
        # print "len of video_list: in twoStreamData",len(self.video_list)
        # my_indexes = list(range(len(self.video_list)))
        # random.shuffle(my_indexes)
        # print "len of my_indexes: in twoStreamData",len(my_indexes)
        self.dynamic_subpath = self.get_video_lists(self.path, self.video_list, 'hmdb51_dynamic/')
        self.frames_subpath = self.get_video_lists(self.path, self.video_list, 'hmdb51_frames/')
        # print "self.dynamic_subpath[1]",self.dynamic_subpath[1]
        self.dynamic = Data(self.name, self.dynamic_subpath, normalization,
                                           sequence_length, crop_size, num_classes, queue_size)
        self.frames = Data(self.name, self.frames_subpath, normalization,
                                           sequence_length, crop_size, num_classes, queue_size)
    
    def get_video_lists(self, path, video_list, type_path):
        lines = video_list
        new_lines = [os.path.join(self.path, type_path, line) for line in lines]
        return new_lines