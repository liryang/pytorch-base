import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
import math
from PIL import Image
import cv2
import time

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])


class Data(Dataset):
    def __init__(self, path, inputsize):
        filenames = next(os.walk(path))[2]
        num_file = len(filenames)

        # Initialize images and labels.
        self.images = np.zeros((num_file, inputsize[0], inputsize[1], 3), dtype=np.uint8)
        self.labels = np.zeros(num_file, dtype=np.int64)
        for index, filename in enumerate(filenames):
            # Read single image and resize it to your expected size
            img = Image.open(os.path.join(path, filename)).convert('RGB')
            img = img.resize(inputsize)
            self.images[index] = img

            if filename[0:3] == 'cat':
                self.labels[index] = int(0)
            else:
                self.labels[index] = int(1)

            if index % 1000 == 0:
                print("Reading the %sth image" % index)

        print("Train Data read finish")

    def __getitem__(self, index):
        img = Image.fromarray(np.uint8(self.images[index]))
        img = transform_train(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)


# 训练
def train(net, train_dataset, criterion, optimizer, batch_size, num_epochs, savepath=None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    net.train()
    per = math.ceil(float(len(train_loader)/100))
    ave_loss = [0] * num_epochs
    ave_acc = [0] * num_epochs
    interval = [0] * num_epochs
    for epoch in range(num_epochs):
        print('current epoch = %d' % (epoch+1))
        time_start = time.time()
        for batch_idx, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
            else:
                images = Variable(images)
                labels = Variable(labels)
            optimizer.zero_grad()
            # forward
            out = net(images)
            _, pred = torch.max(out, 1)
            num_correct = (pred == labels).sum()
            ave_acc[epoch] += num_correct.item()
            loss = criterion(out, labels)
            ave_loss[epoch] += loss.item()
            # backward
            loss.backward()
            optimizer.step()
            if batch_idx % per == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}  Acc: {}/{}'.format(
                    epoch+1, batch_idx * batch_size, len(train_dataset),
                    100 * batch_idx / len(train_loader), loss.item(), num_correct, batch_size))
        ave_loss[epoch] = ave_loss[epoch]/len(train_loader)
        ave_acc[epoch] = ave_acc[epoch]/len(train_dataset)
        interval[epoch] = time.time() - time_start
        print('Train Epoch: {}  Time: {:0.2f}s  Loss: {:.6f}  Acc: {:.6f}'.format(epoch+1, interval[epoch],
                                                                                ave_loss[epoch], ave_acc[epoch]))

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if savepath is not None:
            torch.save(net.state_dict(), './checkpoint/' + net.name + '_' + savepath)
        # else:
            # t7文件是沿用torch7中读取模型权重的方式。而pth文件是python中存储文件的常用格式。
            # torch.save(net.state_dict(), './checkpoint/net_params.pth')
    for epoch in range(num_epochs):
        print('Train Epoch: {}  Time: {:0.2f}s  Loss: {:.6f}  Acc: {:.6f}'.format(epoch + 1, interval[epoch],
                                                                              ave_loss[epoch], ave_acc[epoch]))


# 预测
def test(net, test_dataset, criterion, batch_size):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    net = net.eval()
    eval_loss = 0
    eval_acc = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
            else:
                images = Variable(images)
                labels = Variable(labels)
            out = net(images)
            loss = criterion(out, labels)

            eval_loss += loss.data.item() * labels.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == labels).sum()
            eval_acc += num_correct.data.item()

    print('Test Loss: {:.6f}  Acc: {:.6f}'.format(eval_loss / (len(test_dataset)),
                                                 eval_acc / (len(test_dataset))))

def testsolo(net, filepath):
    net.eval()
    if os.path.isfile(filepath):
        im = cv2.imread(filepath)
        img = cv2.resize(im, (224, 224))
        img = transform_test(img)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
        out = net(img)
        score, pred = torch.max(out, 1)
        if(float(pred) == 0):
            name = 'cat'
        else:
            name = 'dog'
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, name + ':' + str(round(float(score), 2)), (10, 20), font, 0.5, (255, 255, 0), 1)
        cv2.imshow('1', im)
        cv2.waitKey(0)
    elif os.path.isdir(filepath):
        filenames = np.random.permutation(next(os.walk(filepath))[2])
        for index, filename in enumerate(filenames):
            im = cv2.imread(filepath + '/' + filename)
            img = cv2.resize(im, (224, 224))
            img = transform_test(img)
            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img)
            if torch.cuda.is_available():
                img = Variable(img).cuda()
            out = net(img)
            score, pred = torch.max(out, 1)
            if (float(pred) == 0):
                name = 'cat'
            else:
                name = 'dog'
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(im, name + ':' + str(round(float(score), 2)), (10, 20), font, 0.5, (255, 255, 0), 1)
            cv2.imshow('1', im)
            cv2.waitKey(0)





