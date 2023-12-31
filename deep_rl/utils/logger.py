#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from tensorboardX import SummaryWriter
import os
import numpy as np
import torch
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
from .misc import *

def get_logger(name='MAIN', file_name=None, log_dir='./log', skip=False, level=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if file_name is not None:
        file_name = '%s-%s' % (file_name, get_time_str())
        fh = logging.FileHandler('%s/%s.txt' % (log_dir, file_name))
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        fh.setLevel(level)
        logger.addHandler(fh)
    return Logger(log_dir, logger, skip)

class Logger(object):
    def __init__(self, log_dir, vanilla_logger, skip=False):
        try:
            for f in os.listdir(log_dir):
                if not f.startswith('events'):
                    continue
                os.remove('%s/%s' % (log_dir, f))
        except IOError:
            try:
                os.mkdir(log_dir)
            except IOError:
                print('Check log folder...')
        if not skip:
            self.writer = SummaryWriter(log_dir)
        self.info = vanilla_logger.info
        self.debug = vanilla_logger.debug
        self.warning = vanilla_logger.warning
        self.skip = skip
        self.all_steps = {}
        self.log_dir = log_dir

    def to_numpy(self, v):
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        return v

    def get_step(self, tag):
        if tag not in self.all_steps:
            self.all_steps[tag] = 0
        step = self.all_steps[tag]
        self.all_steps[tag] += 1
        return step

    def scalar_summary(self, tag, value, step=None):
        if self.skip:
            return
        self.to_numpy(value)
        if step is None:
            step = self.get_step(tag)
        if np.isscalar(value):
            value = np.asarray([value])
        self.writer.add_scalar(tag, value, step)

    def histo_summary(self, tag, values, step=None):
        if self.skip:
            return
        self.to_numpy(values)
        if step is None:
            step = self.get_step(tag)
        self.writer.add_histogram(tag, values, step, bins=1000)

    def image_summary(self, tag, images, step=None):
        """Log a list of images."""
        if step is None:
            step = self.get_step(tag)
        self.writer.add_image(tag, images, step)
        #img_summaries = []
    #    for i, img in enumerate(images):
            # Write the image to a string
    ##        try:
    #            s = StringIO()
    #        except:
    #            s = BytesIO()
    #        scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
    #        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
    #                                   height=img.shape[0],
    #                                   width=img.shape[1])
    #        self.writer.add_image(tag, images)
            # Create a Summary value
        #    img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        #summary = tf.Summary(value=img_summaries)
        #self.writer.add_image(summary, step)
