'''
From https://github.com/AruniRC/resnet-face-pytorch#pytorch-resnet-on-vggface2
modified by Lei Jiang
add VGG and ResNet architecture
'''
# import fcn
import os.path as osp
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F



class NormFeat(nn.Module):
    ''' L2 normalization of features '''
    def __init__(self, scale_factor=1.0):
        super(NormFeat, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return self.scale_factor * F.normalize(input, p=2, dim=1)


class ScaleFeat(nn.Module):
# https://discuss.pytorch.org/t/is-scale-layer-available-in-pytorch/7954/6?u=arunirc
    def __init__(self, scale_factor=50.0):
        super().__init__()
        self.scale = scale_factor

    def forward(self, input):
        return input * self.scale


# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()



class FCN32sColor(nn.Module):

    def __init__(self, n_class=32, bin_type='one-hot', batch_norm=True):
        super(FCN32sColor, self).__init__()
        self.n_class = n_class
        self.bin_type = bin_type
        self.batch_norm = batch_norm

        # conv1
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv1_2_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv2_2_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_3_bn = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_3_bn = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_3_bn = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        if batch_norm:
            self.fc6_bn = nn.BatchNorm2d(4096)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc7_bn = nn.BatchNorm2d(4096)
        self.drop7 = nn.Dropout2d()

        if bin_type == 'one-hot':
            # NOTE: *two* output prediction maps for hue and chroma
            # TODO - not implemented error should be raised for this!
            self.score_fr_hue = nn.Conv2d(4096, n_class, 1)
            self.upscore_hue = nn.ConvTranspose2d(n_class, n_class, 64, stride=32,
                                              bias=False)
            self.score_fr_chroma = nn.Conv2d(4096, n_class, 1)
            self.upscore_chroma = nn.ConvTranspose2d(n_class, n_class, 64, stride=32,
                                              bias=False)
            self.upscore_hue.weight.requires_grad = False
            self.upscore_chroma.weight.requires_grad = False
        elif bin_type == 'soft':
            self.score_fr = nn.Conv2d(4096, n_class, 1)
            self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32,
                                              bias=False)            
            self.upscore.weight.requires_grad = False # fix bilinear upsampler

        self._initialize_weights()
        # TODO - init from pre-trained network

        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass # leave the default PyTorch init
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


    def forward(self, x):
        h = x
        h = self.conv1_1(h)
        if self.batch_norm:
            h = self.conv1_1_bn(h)
        h = self.relu1_1(h)
        h = self.conv1_2(h)
        if self.batch_norm:
            h = self.conv1_2_bn(h)
        h = self.relu1_2(h)
        h = self.pool1(h)

        if self.batch_norm:
            h = self.relu2_1(self.conv2_1_bn(self.conv2_1(h)))
        else:
            h = self.relu2_1(self.conv2_1(h))
        if self.batch_norm:
            h = self.relu2_2(self.conv2_2_bn(self.conv2_2(h)))
        else:
            h = self.relu2_2(self.conv2_2_bn(self.conv2_2(h)))
        h = self.pool2(h)

        if self.batch_norm:
            h = self.relu3_1(self.conv3_1_bn(self.conv3_1(h)))
        else:
            h = self.relu3_1(self.conv3_1(h))
        if self.batch_norm:
            h = self.relu3_2(self.conv3_2_bn(self.conv3_2(h)))
        else:
            h = self.relu3_2(self.conv3_2(h))
        if self.batch_norm:
            h = self.relu3_3(self.conv3_3_bn(self.conv3_3(h)))
        else:
            h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        if self.batch_norm:
            h = self.relu4_1(self.conv4_1_bn(self.conv4_1(h)))
        else:
            h = self.relu4_1(self.conv4_1(h))
        if self.batch_norm:
            h = self.relu4_2(self.conv4_2_bn(self.conv4_2(h)))
        else:
            h = self.relu4_2(self.conv4_2(h))
        if self.batch_norm:
            h = self.relu4_3(self.conv4_3_bn(self.conv4_3(h)))
        else:
            h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        if self.batch_norm:
            h = self.relu5_1(self.conv5_1_bn(self.conv5_1(h)))
        else:
            h = self.relu5_1(self.conv5_1(h))
        if self.batch_norm:
            h = self.relu5_2(self.conv5_2_bn(self.conv5_2(h)))
        else:
            h = self.relu5_2(self.conv5_2(h))
        if self.batch_norm:
            h = self.relu5_3(self.conv5_3_bn(self.conv5_3(h)))
        else:
            h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        if self.batch_norm:
            h = self.relu6(self.fc6_bn(self.fc6(h)))
        else:
            h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        if self.batch_norm:
            h = self.relu7(self.fc7_bn(self.fc7(h)))
        else:
            h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        if self.bin_type == 'one-hot':
            # hue prediction map
            h_hue = self.score_fr_hue(h)
            h_hue = self.upscore_hue(h_hue)
            h_hue = h_hue[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

            # chroma prediction map
            h_chroma = self.score_fr_chroma(h)
            h_chroma = self.upscore_chroma(h_chroma)
            h_chroma = h_chroma[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
            h = (h_hue, h_chroma)

        elif self.bin_type == 'soft':
            h = self.score_fr(h)
            h = self.upscore(h)
            h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h




class FCN16sColor(nn.Module):

    def __init__(self, n_class=32, bin_type='one-hot', batch_norm=True):
        super(FCN16sColor, self).__init__()
        self.n_class = n_class
        self.bin_type = bin_type
        self.batch_norm = batch_norm

        # conv1
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv1_2_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv2_2_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_3_bn = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_3_bn = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_3_bn = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        if batch_norm:
            self.fc6_bn = nn.BatchNorm2d(4096)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc7_bn = nn.BatchNorm2d(4096)
        self.drop7 = nn.Dropout2d()

        if bin_type == 'one-hot':
            # NOTE: *two* output prediction maps for hue and chroma
            raise NotImplementedError('TODO - FCN 16s for separate hue-chroma')
        elif bin_type == 'soft':
            self.score_fr = nn.Conv2d(4096, n_class, 1)
            self.score_pool4 = nn.Conv2d(512, n_class, 1)

            self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2,
                                              bias=False) 
            self.upscore16 = nn.ConvTranspose2d(n_class, n_class, 32, stride=16,
                                              bias=False)            
            self.upscore2.weight.requires_grad = False # fix bilinear upsamplers
            self.upscore16.weight.requires_grad = False

        self._initialize_weights()
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass # leave the default PyTorch init
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


    def forward(self, x):
        h = x
        h = self.conv1_1(h)
        if self.batch_norm:
            h = self.conv1_1_bn(h)
        h = self.relu1_1(h)
        h = self.conv1_2(h)
        if self.batch_norm:
            h = self.conv1_2_bn(h)
        h = self.relu1_2(h)
        h = self.pool1(h)

        if self.batch_norm:
            h = self.relu2_1(self.conv2_1_bn(self.conv2_1(h)))
        else:
            h = self.relu2_1(self.conv2_1(h))
        if self.batch_norm:
            h = self.relu2_2(self.conv2_2_bn(self.conv2_2(h)))
        else:
            h = self.relu2_2(self.conv2_2_bn(self.conv2_2(h)))
        h = self.pool2(h)

        if self.batch_norm:
            h = self.relu3_1(self.conv3_1_bn(self.conv3_1(h)))
        else:
            h = self.relu3_1(self.conv3_1(h))
        if self.batch_norm:
            h = self.relu3_2(self.conv3_2_bn(self.conv3_2(h)))
        else:
            h = self.relu3_2(self.conv3_2(h))
        if self.batch_norm:
            h = self.relu3_3(self.conv3_3_bn(self.conv3_3(h)))
        else:
            h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        if self.batch_norm:
            h = self.relu4_1(self.conv4_1_bn(self.conv4_1(h)))
        else:
            h = self.relu4_1(self.conv4_1(h))
        if self.batch_norm:
            h = self.relu4_2(self.conv4_2_bn(self.conv4_2(h)))
        else:
            h = self.relu4_2(self.conv4_2(h))
        if self.batch_norm:
            h = self.relu4_3(self.conv4_3_bn(self.conv4_3(h)))
        else:
            h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        if self.batch_norm:
            h = self.relu5_1(self.conv5_1_bn(self.conv5_1(h)))
        else:
            h = self.relu5_1(self.conv5_1(h))
        if self.batch_norm:
            h = self.relu5_2(self.conv5_2_bn(self.conv5_2(h)))
        else:
            h = self.relu5_2(self.conv5_2(h))
        if self.batch_norm:
            h = self.relu5_3(self.conv5_3_bn(self.conv5_3(h)))
        else:
            h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        if self.batch_norm:
            h = self.relu6(self.fc6_bn(self.fc6(h)))
        else:
            h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        if self.batch_norm:
            h = self.relu7(self.fc7_bn(self.fc7(h)))
        else:
            h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        if self.bin_type == 'one-hot':
            raise NotImplementedError('TODO - FCN 16s for separate hue-chroma')
        elif self.bin_type == 'soft':
            h = self.score_fr(h)
            h = self.upscore2(h)
            upscore2 = h  # 1/16

            h = self.score_pool4(pool4)
            h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
            score_pool4c = h # 1/16

            h = upscore2 + score_pool4c
            
            h = self.upscore16(h)
            h = h[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]].contiguous()

        return h


    def copy_params_from_fcn32s(self, fcn32s):
        for name, l1 in fcn32s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data.copy_(l1.weight.data)
            l2.bias.data.copy_(l1.bias.data)



class FCN8sColor(nn.Module):

    def __init__(self, n_class=32, bin_type='one-hot', batch_norm=True):
        super(FCN8sColor, self).__init__()
        self.n_class = n_class
        self.bin_type = bin_type
        self.batch_norm = batch_norm

        # conv1
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv1_2_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv2_2_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_3_bn = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_3_bn = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_3_bn = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        if batch_norm:
            self.fc6_bn = nn.BatchNorm2d(4096)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc7_bn = nn.BatchNorm2d(4096)
        self.drop7 = nn.Dropout2d()

        if bin_type == 'one-hot':
            # NOTE: *two* output prediction maps for hue and chroma
            raise NotImplementedError('TODO - FCN 16s for separate hue-chroma')
        elif bin_type == 'soft':
            self.score_fr = nn.Conv2d(4096, n_class, 1)
            self.score_pool3 = nn.Conv2d(256, n_class, 1)
            self.score_pool4 = nn.Conv2d(512, n_class, 1)

            self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2,
                                              bias=False) 
            self.upscore8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8,
                                              bias=False)  
            self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2,
                                              bias=False) 

            self.upscore2.weight.requires_grad = False # fix bilinear upsamplers
            self.upscore8.weight.requires_grad = False
            self.upscore_pool4.weight.requires_grad = False

        self._initialize_weights()
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass # leave the default PyTorch init
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


    def forward(self, x):
        h = x
        h = self.conv1_1(h)
        if self.batch_norm:
            h = self.conv1_1_bn(h)
        h = self.relu1_1(h)
        h = self.conv1_2(h)
        if self.batch_norm:
            h = self.conv1_2_bn(h)
        h = self.relu1_2(h)
        h = self.pool1(h)

        if self.batch_norm:
            h = self.relu2_1(self.conv2_1_bn(self.conv2_1(h)))
        else:
            h = self.relu2_1(self.conv2_1(h))
        if self.batch_norm:
            h = self.relu2_2(self.conv2_2_bn(self.conv2_2(h)))
        else:
            h = self.relu2_2(self.conv2_2_bn(self.conv2_2(h)))
        h = self.pool2(h)

        if self.batch_norm:
            h = self.relu3_1(self.conv3_1_bn(self.conv3_1(h)))
        else:
            h = self.relu3_1(self.conv3_1(h))
        if self.batch_norm:
            h = self.relu3_2(self.conv3_2_bn(self.conv3_2(h)))
        else:
            h = self.relu3_2(self.conv3_2(h))
        if self.batch_norm:
            h = self.relu3_3(self.conv3_3_bn(self.conv3_3(h)))
        else:
            h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        if self.batch_norm:
            h = self.relu4_1(self.conv4_1_bn(self.conv4_1(h)))
        else:
            h = self.relu4_1(self.conv4_1(h))
        if self.batch_norm:
            h = self.relu4_2(self.conv4_2_bn(self.conv4_2(h)))
        else:
            h = self.relu4_2(self.conv4_2(h))
        if self.batch_norm:
            h = self.relu4_3(self.conv4_3_bn(self.conv4_3(h)))
        else:
            h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        if self.batch_norm:
            h = self.relu5_1(self.conv5_1_bn(self.conv5_1(h)))
        else:
            h = self.relu5_1(self.conv5_1(h))
        if self.batch_norm:
            h = self.relu5_2(self.conv5_2_bn(self.conv5_2(h)))
        else:
            h = self.relu5_2(self.conv5_2(h))
        if self.batch_norm:
            h = self.relu5_3(self.conv5_3_bn(self.conv5_3(h)))
        else:
            h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        if self.batch_norm:
            h = self.relu6(self.fc6_bn(self.fc6(h)))
        else:
            h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        if self.batch_norm:
            h = self.relu7(self.fc7_bn(self.fc7(h)))
        else:
            h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        if self.bin_type == 'one-hot':
            raise NotImplementedError('TODO - FCN 16s for separate hue-chroma')
        elif self.bin_type == 'soft':
            h = self.score_fr(h)
            h = self.upscore2(h)
            upscore2 = h  # 1/16

            h = self.score_pool4(pool4)
            h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
            score_pool4c = h  # 1/16

            h = upscore2 + score_pool4c  # 1/16
            h = self.upscore_pool4(h)
            upscore_pool4 = h # 1/8
            
            h = self.score_pool3(pool3)
            h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
            score_pool3c = h  # 1/8

            h = upscore_pool4 + score_pool3c # 1/8

            h = self.upscore8(h)
            h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h


    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)

def vgg16(pretrained=False, **kwargs):
    model = VGG(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))
    return model


class VGG(nn.Module):
    def __init__(self, number_classes=2000, model_path="model.pkl"):
        super(VGG, self).__init__()
        self.conv11 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, number_classes)

        self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv21(x)
        x = self.relu(x)
        x = self.conv22(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv31(x)
        x = self.relu(x)
        x = self.conv32(x)
        x = self.relu(x)
        x = self.conv33(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv41(x)
        x = self.relu(x)
        x = self.conv42(x)
        x = self.relu(x)
        x = self.conv43(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv51(x)
        x = self.relu(x)
        x = self.conv52(x)
        x = self.relu(x)
        x = self.conv53(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))
    return model


class ResNet(nn.Module):
    """
    block: A sub module
    """

    def __init__(self, layers, num_classes=1000, model_path="model.pkl"):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.modelPath = model_path
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stack1 = self.make_stack(64, layers[0])
        self.stack2 = self.make_stack(128, layers[1], stride=2)
        self.stack3 = self.make_stack(256, layers[2], stride=2)
        self.stack4 = self.make_stack(512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        # initialize parameters
        self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def make_stack(self, planes, blocks, stride=1):
        downsample = None
        layers = []

        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out





