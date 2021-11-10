import torch
import os
from PIL import Image
from torch.utils.data import Dataset


class MyMnistDataset(Dataset):
    def __init__(self, root, transform):

        self.myMnistPath = root
        self.imagesData = []
        self.labelsData = []
        self.labelsDict = {}
        self.trans = transform

        self.loadLabelsDate()
        self.loadImageData()

    # 读取标签txt文件，并生成字典
    def loadLabelsDate(self):
        labelsPath = os.path.join(self.myMnistPath, "labels", "labels.txt")
        f = open(labelsPath)
        lines = f.readlines()
        for line in lines:
            name = line.split(' ')[0]
            label = line.split(' ')[1]
            self.labelsDict[name] = int(label)

    # 读取手写图片数据，并将图片数据和对应的标签组合在一起
    def loadImageData(self):
        imagesFolderPath = os.path.join(self.myMnistPath, 'images')
        imageFiles = os.listdir(imagesFolderPath)

        for imageName in imageFiles:
            imagePath = os.path.join(imagesFolderPath, imageName)
            image = Image.open(imagePath)
            grayImage = image.convert("L")

            imageTensor = self.trans(grayImage)
            self.imagesData.append(imageTensor)

            self.labelsData.append(self.labelsDict[imageName])

        self.labelsData = torch.Tensor(self.labelsData)

    # 重写魔法函数
    def __getitem__(self, index):
        return self.imagesData[index], self.labelsData[index]

    # 重写魔法函数
    def __len__(self):
        return len(self.labelsData)

