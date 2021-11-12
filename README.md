# 使用教程

代码下载地址：https://gitee.com/martin64/mnist-pytorch

**模型训练**

运行CNN文件下的train.py文件训练卷积神经网络模型，训练完成后会生成文件cnn_trained_model.pth

运行FC文件下的train.py文件训练全连接神经网络模型，训练完成后会生成文件fc_trained_model.pth



**模型测试**

1. 在my_mnist_dateset/classify文件夹下的10个文件夹下放入对应的手写数字图片，图片长和宽随意，注意图片要是白底黑字的。
2. 运行make_ours_dataset.py，它会自动将白底黑字图片转换为黑底白字，并自动生成标签。
3. 如果要测试训练好的CNN模型，请运行CNN文件夹下的trained_model_test.py
4. 如果要测试训练好的FC模型，请运行FC文件下的trained_model_test.py



**模型在训练过程中会自动显示训练进度，有无cuda均可以训练。**

# MNIST介绍

MNIST数据集来自美国国家标准与技术研究所，National Institute of Standards and Technology(NIST). 训练集由250个不同人手写的数字构成，其中50%是高中学生，50%来自人户普查局的工作人员。测试集也是同样比例的手写数字数据。

MNIST数据集分为2个部分，分别含有6000张训练图片和1000张测试图片。

每一张图片图片的大小都是28×28，而且图片的背景色为黑色，字迹为白色。原始图像如下图：

![mnistOriginImages](https://gitee.com/martin64/mnist-pytorch/raw/master/images/mnistOriginImages.jpg)

如果是用pytorch，我们可以用下面的代码来下载MNIST数据集。

```python
from torchvision import datasets

#训练数据集，torchvision中封装了mnist数据集的下载方式，调用下面函数就会自动下载
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True)

#测试数据集
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True)
```

下载后，在你当前项目下的相对路径../dataset/mnist文件夹下就会有mnist数据集的下载文件了。

**注意：../的意思是当前项目文件夹的上一层，不熟悉的可以补一下绝对路径和相对路径的知识。**

**其中**

- train_images：训练集图片
- train_labels：训练集标签
- t10k_images：测试集图片
- t10k_labels：测试集标签

![mnistSet](https://gitee.com/martin64/mnist-pytorch/raw/master/images/mnistSet.jpg)

直接下载下来的数据是无法通过解压或者应用程序打开的，因为这些文件不是任何标准的图像格式而是以字节的形式存储的，如果你想看具体的MNIST数据集图像，可以参考下面的链接：

[MNIST数据集读取](https://blog.csdn.net/panrenlong/article/details/81736754?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162738630716780274134250%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=162738630716780274134250&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-81736754.first_rank_v2_pc_rank_v29&utm_term=mnist%E8%AF%BB%E5%8F%96&spm=1018.2226.3001.4187)

# 训练模型

## 全连接神经网络

全连接神经网络特点是在隐藏层的每一层中，上一层的每一个神经元都与下一层所有神经元相连。一个神经网络的示例如下：

![FCExample](https://gitee.com/martin64/mnist-pytorch/raw/master/images/FCExample.jpg)

神经网络的训练主要是前向传播后计算损失函数，然后反向传播，依次更新权重。

### 数据预处理

MNIST数据集中的图像像素值在0~255之间，但是神经网络希望输入的数据介于0-1之间，并且服从正态分布，所以我们要用transform将输入数据进行标准化。

[为什么要进行标准化（Normalization）](https://www.zhihu.com/column/c_141391545)

- ToTensor将数据转换为张量
- Normalize将数据标准化
- 其中0.1307是均值，0.3081是方差，这两个数据是经验值。

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```

### 构造神经网络

MNIST数据集中的图片都是28×28大小的，而且是灰度图。而全连接神经网络的输入要是一个行向量，所以我们要把28×28的矩阵转换成28×28=764的行向量，作为神经网络的输入。

```
x = x.view(-1, 784);
```

后面的依次是512,256,128,64,的线性层，由于我们是在做一个多分类问题，而且预测的值是0-9中的一个，所以最后的输出应该是一个1×10的行向量。具体网络结构如下图：

![FCMnist](https://gitee.com/martin64/mnist-pytorch/raw/master/images/FCMnist.jpg)

全连接神经网络模型代码如下：

```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__();
        self.l1 = torch.nn.Linear(784, 512);
        self.l2 = torch.nn.Linear(512, 256);
        self.l3 = torch.nn.Linear(256, 128);
        self.l4 = torch.nn.Linear(128, 64);
        self.l5 = torch.nn.Linear(64, 10);
        self.relu = torch.nn.ReLU();

    def forward(self, x):
        x = x.view(-1, 784);
        x = self.l1(x);
        x = self.relu(x);

        x = self.l2(x);
        x = self.relu(x);

        x = self.l3(x);
        x = self.relu(x);

        x = self.l4(x);
        x = self.relu(x);

        x = self.l5(x);

        return x;
```

### 训练模型

只要运行项目中的**MNIST/FC/train.py**代码就能训练神经网络了。

我们知道对于多分类问题，是用softmax函数，然后用one-hot编码来计算loss，但是上面的神经网络forward函数中并没有出现softmax函数，原因是在下面这行代码中，CrossEntropyLoss已经包含了softmax函数。
$$
CrossEntropyLoss=LogSoftmax+NLLLoss.
$$

```python
criterion = torch.nn.CrossEntropyLoss();
```

也就是说使用CrossEntropyLoss最后一层(线性层)是不需要做其他变化的；使用NLLLoss之前，需要对最后一层(线性层)先进行SoftMax处理，再进行log操作。 

 accurancy on test set:76 %
accurancy on test set:90 %
accurancy on test set:92 %
accurancy on test set:94 %
accurancy on test set:95 %
accurancy on test set:96 %
accurancy on test set:96 %
accurancy on test set:96 %
accurancy on test set:96 %
accurancy on test set:96 %

上面是在我电脑上运行后的准确率，如果你的准确率没那么高，可以在增加训练的次数。

### 训练后模型的保存

训练完成后，我们可以保存训练好的模型，所以在FCMnist.py中最后一行加上了这样一行代码：

```python
torch.save(model.state_dict(),"fc_trained_model.pth")
```

这样，在**MNIST/FC**文件夹下，就会有一个**fc_trained_model.pth**文件。

**数据下载，训练，模型保存都在MNIST/FC/train.py代码中实现了，所以不需要改动任何代码，只要运行它就行了。**

## 卷积神经网络

通过上一节我们知道，全连接神经网络需要将输入的图像数据矩阵转换成一个行向量，但是这样做有缺点，那就是：

**图像的数据是一个矩阵，也就是一个像素点的和它所在位置的上下左右像素点有很大的相关性，将像素矩阵flatten后，左右的位置关系保留了，但是上下的位置关系却被破坏了。**

**卷积神经网络和全连接神经网络的不同之处在于，卷积神经网络的输入是图像的原始矩阵，这样保留了图像的上下左右位置关系。**

关于CNN神经网络入门，我推荐知乎的一个优秀答主[蒋竺波](https://www.zhihu.com/people/followbobo)，下面是他的CNN入门文章，大家也可以去他的主页查看。

[什么是卷积（Convolution）](https://zhuanlan.zhihu.com/p/30994790)

[卷积层是如何提取特征的](https://zhuanlan.zhihu.com/p/31657315)

[什么是采样层（pooling）](https://zhuanlan.zhihu.com/p/32299939)

[什么是激活函数（Activation Function）](https://zhuanlan.zhihu.com/p/32824193)

[什么是全连接层（Full Connected Layer）](https://zhuanlan.zhihu.com/p/33841176)

[什么是Softmax](https://www.zhihu.com/column/c_141391545)

关于损失函数：

[什么是交叉熵（Cross Entropy）](https://www.bilibili.com/video/BV15V411W7VB)

### 数据预处理

过程和全连接神经网络一样，就不介绍了

### 构造神经网络

关于MNIST的卷积神经网络是这样设计的，

1. 第一层是一个卷积层，输入通道是1，输出通道是10，卷积核大小是5×5。输入维度是1×28×28，输出维度是10×24×24。
2. 第二层是一个下采样层，采样核大小是2×2，输入维度是10×24×24，输出维度是10×12×12。

3. 第三层是一个卷积层，输入通道是10，输出通道是20，卷积核大小是5×5。输入维度是10×12×12，输出维度是20×8×8。
4. 第四层是一个下采样层，采样核大小是2×2，输入维度是20×8×8，输出维度是20×4×4。
5. 第五层是flatten层，大小为320
6. 第六层是一个全连接层

整个神经网络的结构如下图所示(全连接神经网络部分由于输入320个输入太多，所以用16代替了)。

<img src="https://gitee.com/martin64/mnist-pytorch/raw/master/images/CNN.jpg" alt="CNN" style="zoom: 80%;" />

<img src="https://gitee.com/martin64/mnist-pytorch/raw/master/images/CNN_FC.jpg" alt="CNN_FC" style="zoom: 25%;" />

卷积神经网络模型的代码：

```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__();
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5);
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5);
        self.pooling=torch.nn.MaxPool2d(2);
        self.fc=torch.nn.Linear(320,10);
        self.relu=torch.nn.ReLU();

    def forward(self, x):
        batch_size=x.size(0);

        x=self.conv1(x);
        x=self.pooling(x);
        x=self.relu(x);

        x = self.conv2(x);
        x = self.pooling(x);
        x = self.relu(x);

        x=x.view(batch_size,-1);
        x=self.fc(x);

        return x;
```

### 训练模型和模型保存

只要运行**CNNMnist.py**就可以训练卷积神经网络了，然后会在当前路径的文件下生成一个**CNNMnist.pth**，这样就保存好了训练好的模型。

# 制作自己的数据集

[python制作自己的数据集](https://blog.csdn.net/zhangjunp3/article/details/79627824?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162772919416780274116746%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=162772919416780274116746&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-79627824.first_rank_v2_pc_rank_v29&utm_term=python%E5%88%B6%E4%BD%9C%E8%87%AA%E5%B7%B1%E7%9A%84%E6%95%B0%E6%8D%AE%E9%9B%86&spm=1018.2226.3001.4187)

在当前项目下新建一个文件夹，命名为my_mnist_dateset,在my_mnist_dateset文件夹下新建3个文件夹，classify、images、labels。

- classify文件夹下有10个文件夹，保存未处理的手写图片，比如1文件夹保存数字1的手写图片，依次类推。
- image文件夹保存从classify文件夹中读取的所有图片。
- labels文件夹保存标签。

整个目录树如图所示。

![contensTree](https://gitee.com/martin64/mnist-pytorch/raw/master/images/contentsTree.jpg)

## 手写图片制作

打开画板，写一些数字，然后用图片编辑软件裁剪图片，使得数字大概在裁剪图片的中心。

我手写的10个数字如下：

![](https://gitee.com/martin64/mnist-pytorch/raw/master/images/digitaContens.jpg)

## 图片批处理与标签生成

由于MNIST的标准图片是28×28的，所以如果要进行测试，我们自己制作的图片也要是28×28的。我们可以运行**make_ours_dataset.py**进行批处理。这样就会把image文件下的图片都更改成28×28大小的图片。

下图是图片批处理后的效果。

![myHandWriting](https://gitee.com/martin64/mnist-pytorch/raw/master/images/digitHandWriting.jpg)



同时，在labels文件夹下会自动生成labels.txt文件，里面保存了图片名称和对应的标签，如下图所示：

![labels](https://gitee.com/martin64/mnist-pytorch/raw/master/images/labels.jpg)



# 模型测试

在训练神模型那一节，我们训练了两个神经网络，FC和CNN，并保存了训练好的模型，也就是**FCMnist.pth**与**CNNMnist.pth**两个文件，这两个文件保留了我们训练好模型的参数。

接下来我们要加载这两个训练好的模型，测试自己的数据集。

## 载入自己的数据集

首先，和前面载入MNIST数据集一样，对于自己的数据集，也要先标准化。

然后就是继承Dataset，写一个读取自己数据集的类，代码如下：

```python
class MyMnistDataset(Dataset):
    def __init__(self, filePath):

        self.myMnistPath = filePath;
        self.imagesData = [];
        self.labelsData = [];
        self.labelsDict = {};

        self.loadLabelsDate();
        self.loadImageData();

    # 读取标签txt文件，并生成字典
    def loadLabelsDate(self):
        labelsPath = os.path.join(self.myMnistPath, "labels", "labels.txt");
        f = open(labelsPath);
        lines = f.readlines();
        for line in lines:
            name = line.split(' ')[0];
            label = line.split(' ')[1];
            self.labelsDict[name] = int(label);

    # 读取手写图片数据，并将图片数据和对应的标签组合在一起
    def loadImageData(self):
        imagesFolderPath = os.path.join(self.myMnistPath, 'images');
        imageFiles = os.listdir(imagesFolderPath);

        for imageName in imageFiles:
            imagePath = os.path.join(imagesFolderPath, imageName);
            image = cv2.imread(imagePath);
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);

            self.imagesData.append(image);
            self.labelsData.append(self.labelsDict[imageName]);

        self.imagesData = torch.Tensor(self.imagesData).unsqueeze(1);
        self.labelsData = torch.Tensor(self.labelsData);

    # 重写魔法函数
    def __getitem__(self, index):
        return self.imagesData[index], self.labelsData[index]

    # 重写魔法函数
    def __len__(self):
        return len(self.labelsData);

#载入自己的数据集
dataset = MyMnistDataset('my_mnist_dateset');
test_loader = DataLoader(dataset=dataset, shuffle=False)
```



## 载入训练好的模型

载入训练好的模型之前，我们需要定义一个和训练好的模型一样的神经网络，这样**FCMnist.pth**与**CNNMnist.pth**这两个文件中的参数才能正确嵌套在神经网络里面。

比如FCTrainedModelTest.py代码是测试全连接神经网络在MyMnist数据集上的效果，**FCTrainedModelTest.py**定义的FCNet和**FCMnist.py**里面的Net结构是一样的，这样才能将**FCMnist.py**训练的模型（**FCMnist.pth**）加载在FCNet之中。

```
class FCNet(torch.nn.Module):
    def __init__(self):
        super(FCNet, self).__init__();
        self.l1 = torch.nn.Linear(784, 512);
        self.l2 = torch.nn.Linear(512, 256);
        self.l3 = torch.nn.Linear(256, 128);
        self.l4 = torch.nn.Linear(128, 64);
        self.l5 = torch.nn.Linear(64, 10);
        self.relu = torch.nn.ReLU();

    def forward(self, x):
        x = x.view(-1, 784);
        x = self.l1(x);
        x = self.relu(x);

        x = self.l2(x);
        x = self.relu(x);

        x = self.l3(x);
        x = self.relu(x);

        x = self.l4(x);
        x = self.relu(x);

        x = self.l5(x);

        return x;
```

运行**FCTrainedModelTest.py**就能加载**FCMnist.pth**,然后测试**FCMnist.pth**在自己数据集上的准确度。

同理运行**CNNTrainedModelTest.py**就能加载**CNNMnist.pth**，然后测试**CNNMnist.pth**在自己数据集上的准确度。

# 相关课程学习

本项目是在B站上学习 刘二大人 的课程 《Pytorch深度学习实践》后完善做成的，相关课程链接如下：

https://www.bilibili.com/video/BV1Y7411d7Ys?from=search&seid=5291537098843647660

两外，CSDN博主 错错莫 将课程中的例子用代码实现了出来，相关链接如下：

https://blog.csdn.net/bit452/category_10569531.html

 

