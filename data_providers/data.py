import cv2
import os
import random
import tempfile
import numpy as np
from Queue import Queue
from threading import Thread


class Data(object):
	"""docstring for Data"""
	def __init__(self, name, path, normalization, sequence_length,crop_size,
		num_classes,queue_size):
		
		self.name = name
		self.path = path
		self.sequence_length = sequence_length
		self.crop_size = crop_size
		self.normalization = normalization
		self.num_classes = num_classes
		self.queue = DataQueue(name,queue_size)
		self.examples = None
		self._start_data_thread()

	def get_frames(self,filename,sequence_length=16):
		video = []
		start = 0
		for parent, dirnames,files in os.walk(filename):
			filenames = [file for file in files if file.endswith(".jpeg",".jpg",".png")]
			if (len(filenames) < sequence_length):
				return	None
			suffix = filenames[0].split('.',1)[1]
			filenames_int = [i.split('.',1)[0] for i in filenames]
			filenames_int = sorted(filenames_int)
			start = random.randint(0,len(filenames) - sequence_length)
			for i in range(start,start+sequence_length):
				img_name = str(filename) + '/' + str(filenames_int[i]) + '.' + suffix
				img = cv2.imread(img_name)
				img = cv2.resize(img,self.crop_size)
				if self.normalization:
					img = self.normalize_image(img,self.normalization)
				video.append(img)
			return video

	def normalize_image(self, img, normalization):
		if normalization = 'std':
			img = (img - np.mean(img))/np.std(img)
		elif normalization = 'divide_256':
			img = img/256
		elif normalization = 'divide_255':
			img = img/255
		else:
			raise Exception("please set the norm method")
		return img

	def get_video_data(self):
		while True:
			index = random.randint(0,len(self.paths)-1)
			video_path,label = self.paths[index].strip('\n').split()
			video = self.get_frames(video_path, self.sequence_length)
			if video is not None and len(video)==self.sequence_length:
				video = np.array(video)
				label = np.array(int(label))
				self.queue.put((video,label))

	def _start_data_thread(self):
		print("start thread: %s data preparation ..." % self.name)
		self.worker = Thread(target=self.get_video_data)
		self.worker = setDaemon(True)
		self.worker.start()

	@property
	def num_examples(self):
		if not self.examples:
			total = 0
			for line in self.paths:
				video_path, _ = line.strip('\n').split()
				for root,dirs,files in os.walk(video_path):
					total += len(files)
			self.examples = total / self.sequence_length
		return self.examples

	def next_batch(self, batch_size):
		videos, labels = self.queue.get(batch_size)
		videos = np.array(videos)
		labels = np.array(labels)
		labels = self.lables_to_one_hot(labels,self.num_classes)
		return videos,labels

	def labels_to_one_hot(self, labels, num_classes):
		# labels = labels
		new_labels = np.zeros((labels.shape[0],num_classes))
		new_labels[range(labels.shape[0]),labels] = np.ones(labels.shape)
		return new_labels
	def labels_for_one_hot(self,labels):
		return np.argmax(labels,axis=1)


class DataQueue(object):
	"""docstring for Data"""
	def __init__(self, name, max_items,block=True):
		self.name = name
		self.block = block
		self.max_items = max_items
		self.queue = Queue(max_items)

	@property
	def queue(self):
		return self.queue
	@property
	def name(self):
		return self.name
	def put(self, data):
		self.queue.put(data,self.block)
	def get(self,batch_size):
		videos = []
		labels = []
		for i in range(batch_size):
			video,label = self.queue.get(self.block)
			videos.append(video)
			labels.append(label)
		return videos,labels
		


class Dataset(object):
	def __init__(self, path, num_classes, validation_set=None, test=False,
		validation_split=None, normalization=None, crop_size=(64,64),
		train_queue=None, valid_queue=None, test_queue=None,
		train=False, queue_size=300, **kwargs):
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

	train_labels = self.get_video_lists(os.path.join(self._path,'train.list'))
	test_labels = self.get_video_lists(os.path.join(self._path,'test.list'))

	if validation_set and validation_split:
		random.shuffle(train_labels)
		valid_labels = train_labels[:validation_split]
		train_labels = train_labels[validation_split:]
		self.validation = Data('validation',valid_labels,normalization,
			sequence_length,crop_size,num_classes,queue_size)

	if train:
		self.train = Data('train',train_labels,normalization,sequence_length,
			crop_size,num_classes,queue_size)

	if test:
		self.test = Data('test',test_labels,normalization,sequence_length,
			crop_size,num_classes,queue_size)

	if validation_set and not validation_split:
		self.validation = Data('validation',test_labels,normalization,sequence_length,
			crop_size,num_classes,queue_size)


	def get_video_lists(self, path):
		lines = open(path, 'r')
		lines = list(lines)
		new_lines = [os.path.join(self._path,line) for line in lines]
		return new_lines

	@property
	def data_shape(self):
		return (self._sequence_length,self._crop_size[1],self._crop_size[0],3)
	@property
	def n_classes(self):
		return self._num_classes


