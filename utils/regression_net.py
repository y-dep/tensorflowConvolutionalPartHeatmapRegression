# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library
# (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from kaffe.tensorflow import Network
import tensorflow as tf
import numpy as np


class RegressionSubnet(Network):

    def setup(self, is_training, n_classes):
        '''Network definition.
        ####Detection Network
        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of 
                       the-pretrained model frozen.
        '''
        # 384x384-->192x192
# A1
        (self.feed('data')
             .conv(9, 9, 64, 1, 1, biased=True, relu=False, name='C1_conv')
             .conv(13, 13, 64, 1, 1, biased=True, relu=False, name='C2_conv')
             .conv(13, 13, 64, 1, 1, biased=True, relu=False, name='C3_conv')
             .conv(15, 15, 64, 1, 1, biased=True, relu=False, name='C4_conv')
             .conv(1, 1, 64, 1, 1, biased=True, relu=False, name='C5_conv')
             .conv(1, 1, 64, 1, 1, biased=True, relu=False, name='C6_conv')
             .conv(1, 1, 64, 1, 1, biased=True, relu=False, name='C7_conv')
             .deconv(380, 380, 8, 4, 16, name='C8_deconv'))


# # A9 mid3
#         (self.feed('add2')
#              .bilinear(name='A9_mid3_bilinear'))

#         (self.feed('A1_pool')
#              .printLayer(name='print_A1_pool'))

#         (self.feed('A2_pool')
#              .printLayer(name='print_A2_pool'))

#         (self.feed('A3_pool')
#              .printLayer(name='print_A3_pool'))

#         (self.feed('A4_pool')
#              .printLayer(name='print_A4_pool'))

#         (self.feed('A5_pool')
#              .printLayer(name='print_A5_pool'))

#         (self.feed('A6_conv')
#              .printLayer(name='print_A6_conv'))

#         (self.feed('A7_conv')
#              .printLayer(name='print_A7_conv'))

#         (self.feed('A8_mid_conv')
#              .printLayer(name='print_A8_mid_conv'))

#         (self.feed('A9_mid1_bilinear')
#              .printLayer(name='print_A9_mid1_bilinear'))

#         (self.feed('A8_up_conv')
#              .printLayer(name='print_A8_up_conv'))

#         (self.feed('A8_down_conv')
#              .printLayer(name='print_A8_down_conv'))

#         (self.feed('add1')
#              .printLayer(name='print_add1'))

#         (self.feed('A9_mid2_bilinear')
#              .printLayer(name='print_A9_mid2_bilinear'))

#         (self.feed('add2')
#              .printLayer(name='print_add2'))

#         (self.feed('A9_mid3_bilinear')
#              .printLayer(name='print_A9_mid3_bilinear'))
