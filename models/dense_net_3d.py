import os
import time
import shutil
import platform
from datetime import timedelta

import numpy as np
import tensorflow as tf

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))

class DenseNet3D(object):
