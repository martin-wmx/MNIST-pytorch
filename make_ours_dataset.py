import os
from PIL import Image
import PIL.ImageOps

classifyFolderPath = "./my_mnist_dateset/classify"
imageFolderPath = "./my_mnist_dateset/images"
labelFolderPath = "my_mnist_dateset/labels/"
labelFilePath = os.path.join(labelFolderPath, "labels.txt")

assert os.path.exists(classifyFolderPath), "file: '{}' dose not exist.".format(classifyFolderPath)

if not os.path.exists(imageFolderPath):
    os.mkdir(imageFolderPath)

if not os.path.exists(labelFolderPath):
    os.mkdir(labelFolderPath)

classes = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

f = open(labelFilePath, 'w')

index = 0
for className in classes:
    classPath = os.path.join(classifyFolderPath, className)
    for imageName in os.listdir(classPath):
        imagePath = os.path.join(classPath, imageName)

        image = Image.open(imagePath)
        inverted_image = PIL.ImageOps.invert(image)

        newName = "{}.jpg".format(index)
        inverted_image.save(os.path.join(imageFolderPath, newName))

        index += 1

        # 在labels.txt文件中写入名称和标签，格式为：名称 空格 标签
        f.writelines(newName + " " + className)
        f.write('\n')

f.close()
