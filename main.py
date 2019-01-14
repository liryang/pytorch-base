import model_alexnet
import model_vgg16
import model_inception_v3
import model_resnet
import model_densenet
from control import Data, train, test, testsolo
import torch
import torchvision
from torch import nn, optim

# 路径
train_path = 'F:/data_sets/kaggle/train'
test_path = 'F:/data_sets/kaggle/test'
param_data = 'params.pth'

# 超参数
batch_size = 16
learing_rate = 1e-4
num_epoches = 3

trainflag = False
testflag = False

if __name__ == '__main__':
    # 导入网络，定义损失函数和优化方法
    # alexnet
    # net = model_alexnet.alexnet(2, first=True)
    # net = model_alexnet.alexnet(2, pthpath=param_data)

    # vgg16
    # net = model_vgg16.vgg16_bn(2, first=True)
    # net = model_vgg16.vgg16_bn(2, pthpath=param_data)

    # inception3
    # net = model_inception_v3.inception_v3(2, first=True)
    # net = model_inception_v3.inception_v3(2, pthpath=param_data)

    # resnet
    # net = model_resnet.resnet34(2, first=True)
    # net = model_resnet.resnet34(2, pthpath=param_data)

    # net = model_resnet.resnet50(2, first=True)
    # net = model_resnet.resnet50(2, pthpath=param_data)

    # densenet
    # net = model_densenet.densenet121(2, first=True)
    net = model_densenet.densenet121(2, pthpath=param_data)

    # net.keep_weight()
    if torch.cuda.is_available():
        net = net.cuda()
    criterion = nn.CrossEntropyLoss()
    if trainflag:
        train_dataset = Data(train_path, net.inputsize)
        optimizer = optim.SGD(net.parameters(), lr=learing_rate)
        train(net, train_dataset, criterion, optimizer, batch_size=batch_size, num_epochs=num_epoches,
              savepath=param_data)
    if testflag:
        test_dataset = Data(test_path, net.inputsize)
        test(net, test_dataset, criterion, batch_size=batch_size)

    testsolo(net, 'F:/data_sets/kaggle/test1')
