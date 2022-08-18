import torch
import numpy as np
from helper import *


class Channels():
    def __init__(self,dimension):
        self.dimension=dimension
    def random_channel(self,att_db):
        '''
        get random channel with random u v and fix attenuation
        :param att: channel attenuation. 1d tensor, in db.
        :return: channel matrix
        '''
        return gen_ch(dimensions=self.dimension,att=db_to_times(att_db))

