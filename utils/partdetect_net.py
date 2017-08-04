# Converted to TensorFlow .caffemodel
# with the DeepLab-ResNet configuration.
# The batch normalisation layer is provided by
# the slim library
# (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim).

from kaffe.tensorflow import Network
import tensorflow as tf
import numpy as np


class PartDetectionSubnet(Network):

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
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='A1_conv_1')#[1,380,380,64]
             .conv(3, 3, 64, 1, 1, biased=True, relu=True, name='A1_conv_2')#[1,380,380,64]
             .max_pool(2, 2, 2, 2, name='A1_pool'))#[1,190,190,64]

# A2
        (self.feed('A1_pool')
             .conv(3, 3, 128, 1, 1, biased=True, relu=True, name='A2_conv_1')
             .conv(3, 3, 128, 1, 1, biased=True, relu=True, name='A2_conv_2')
             .max_pool(2, 2, 2, 2, name='A2_pool'))

# A3
        (self.feed('A2_pool')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='A3_conv_1')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='A3_conv_2')
             .conv(3, 3, 256, 1, 1, biased=True, relu=True, name='A3_conv_3')
             .max_pool(2, 2, 2, 2, name='A3_pool'))

# A4
        (self.feed('A3_pool')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='A4_conv_1')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='A4_conv_2')
             .conv(3, 3, 512, 1, 1, biased=True, relu=True, name='A4_conv_3')
             .max_pool(2, 2, 2, 2, name='A4_pool'))

# A5
        (self.feed('A4_pool')
             .conv(1, 1, 512, 1, 1, biased=True, relu=True, name='A5_conv_1')
             .conv(1, 1, 512, 1, 1, biased=True, relu=True, name='A5_conv_2')
             .conv(1, 1, 512, 1, 1, biased=True, relu=True, name='A5_conv_3')
             .max_pool(2, 2, 2, 2, name='A5_pool'))

# A6
        (self.feed('A5_pool')
             .conv(7, 7, 4096, 1, 1, biased=True, relu=True, name='A6_conv'))

# A7
        (self.feed('A6_conv')
             .conv(1, 1, 4096, 1, 1, biased=True, relu=True, name='A7_conv'))

# A8 mid
        (self.feed('A7_conv')
             .conv(1, 1, 16, 1, 1, biased=True, relu=False, name='A8_mid_conv'))

# A9 mid1
        (self.feed('A8_mid_conv')
             .bilinear(24, 24, name='A9_mid1_bilinear'))

# A8 up
        (self.feed('A3_pool')
             .conv(1, 1, 16, 1, 1, biased=True, relu=False, name='A8_up_conv'))

# A8 down
        (self.feed('A4_pool')
             .conv(1, 1, 16, 1, 1, biased=True, relu=False, name='A8_down_conv'))

# add1= A8_down + A9_mid1
        (self.feed('A9_mid1_bilinear',
                    'A8_down_conv')
             .add(name='add1'))

# A9 mid2
        (self.feed('add1')
             .bilinear(48, 48, name='A9_mid2_bilinear'))

# add2= A8_up + A9_mid2
        (self.feed('A9_mid2_bilinear',
                    'A8_up_conv')
             .add(name='add2'))

# A9 mid3
        (self.feed('add2')
             .bilinear(95, 95, name='A9_mid3_bilinear'))

        # (self.feed('A1_pool')
        #      .printLayer(name='print_A1_pool'))

        # (self.feed('A2_pool')
        #      .printLayer(name='print_A2_pool'))

        # (self.feed('A3_pool')
        #      .printLayer(name='print_A3_pool'))

        # (self.feed('A4_pool')
        #      .printLayer(name='print_A4_pool'))

        # (self.feed('A5_pool')
        #      .printLayer(name='print_A5_pool'))

        # (self.feed('A6_conv')
        #      .printLayer(name='print_A6_conv'))

        # (self.feed('A7_conv')
        #      .printLayer(name='print_A7_conv'))

        # (self.feed('A8_mid_conv')
        #      .printLayer(name='print_A8_mid_conv'))

        # (self.feed('A9_mid1_bilinear')
        #      .printLayer(name='print_A9_mid1_bilinear'))

        # (self.feed('A8_up_conv')
        #      .printLayer(name='print_A8_up_conv'))

        # (self.feed('A8_down_conv')
        #      .printLayer(name='print_A8_down_conv'))

        # (self.feed('add1')
        #      .printLayer(name='print_add1'))

        # (self.feed('A9_mid2_bilinear')
        #      .printLayer(name='print_A9_mid2_bilinear'))

        # (self.feed('add2')
        #      .printLayer(name='print_add2'))

        # (self.feed('A9_mid3_bilinear')
        #      .printLayer(name='print_A9_mid3_bilinear'))
