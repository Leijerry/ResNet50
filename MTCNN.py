import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from MTCNN_Pytorch import simpling
import numpy as np
import os


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            nn.PReLU()
        )

        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv_layer(x)

        cond = F.sigmoid(self.conv4_1(x))
        box_offset = self.conv4_2(x)
        land_offset = self.conv4_3(x)

        return cond, box_offset, land_offset


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # pool1
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # conv3
            nn.PReLU()

        )
        self.line1 = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU()
        )

        self.line2_1 = nn.Linear(128, 1)
        self.line2_2 = nn.Linear(128, 4)
        self.line2_3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.line1(x)

        label = F.sigmoid(self.conv5_1(x))
        box_offset = self.conv5_2(x)
        land_offset = self.conv5_3(x)

        return label, box_offset, land_offset


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # conv1
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # conv2
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool2
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # conv3
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool3
            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # conv4
            nn.PReLU()
        )
        self.line1 = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU()
        )

        self.line2_1 = nn.Linear(256, 1)
        self.line2_2 = nn.Linear(256, 4)
        self.line2_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)

        label = F.sigmoid(self.line2_1(x))
        box_offset = self.line2_2(x)
        land_offset = self.line2_3(x)

        return label, box_offset, land_offset




class MTCNN:

    def __init__(self, train_net, batch_size, data_path, save_model_path, lr=0.001, isCuda=True):

        self.model = train_net
        self.data_path = data_path
        self.batch_size = batch_size
        self.lr = lr
        self.isCuda = isCuda
        self.save_path = save_model_path

        if os.path.exists(self.save_path):
            self.model = torch.load(self.save_path)

        if self.isCuda:
            self.model.cuda()

        self.face_loss = torch.nn.BCELoss()
        self.offset_loss = torch.nn.MSELoss()

        self.opt = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        self.train_net()

    def train_net(self):
        epoch = 1
        IMG_DATA = simpling.FaceDataset(self.data_path)
        for _ in range(10000):
            train_data = data.DataLoader(IMG_DATA, batch_size=self.batch_size, shuffle=True, num_workers=4)
            for train in train_data:

                # img_data ：[512, 3, 24, 24]
                # label ：[512, 1]
                # offset ：[512, 4]
                img_data, label, box_offset, land_offset = train

                if self.isCuda:
                    img_data = img_data.cuda()
                    box_offset = box_offset.cuda()
                    land_offset = land_offset.cuda()

                # P-net
                # face_out : [512, 2, 1, 1]
                # box_offset_out: [512, 4, 1, 1]
                # land_offset_out: [512,10,1,1]
                # R-net、O-net
                # face_out : [512, 2, 1, 1]
                # box_offset_out: [512, 4, 1, 1]
                # land_offset_out: [512,10,1,1]
                face_out, box_offset_out,land_offset_out= self.model(img_data)

                # [512, 2, 1, 1] => [512,2]
                face_out = face_out.squeeze()
                box_offset_out = box_offset_out.squeeze()
                land_offset_out = land_offset_out.squeeze()


                one = torch.ne(label, 2)  # one : torch.Size([512, 1])
                one = one.squeeze()  # one : torch.Size([512]) ： 1,0 int8


                two = torch.ne(label, 0)  # two : [512,1]
                two = two.squeeze()  # two : [512]


                label_10 = label[one]  # [batch,1]
                label_10 = torch.Tensor([self.one_hot(int(i)) for i in label_10.squeeze().numpy()])  # [batch,2]

                face_loss = self.face_loss(face_out[one], label_10.cuda())
                box_offset_loss = self.offset_loss(box_offset_out[two], box_offset[two])
                land_offset_loss = self.offset_loss(land_offset_out[two],land_offset[two])
                self.loss = face_loss + box_offset_loss + land_offset_loss
                self.opt.zero_grad()
                self.loss.backward()
                self.opt.step()
                epoch += 1
                if epoch % 100 == 0:
                    print('Epoch:', epoch, ' Loss：', self.loss.cpu().item())
                    torch.save(self.model, self.save_path)

    def one_hot(self, data):
        hot = np.zeros([2])
        hot[data] = 1
        return hot


if __name__ == '__main__':
    pass


