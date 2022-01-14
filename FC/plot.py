import pandas as pd
import matplotlib.pyplot as plt

csv_path = './train.csv'
data = pd.read_csv(csv_path)

epoch = data['epoch']
loss = data['train loss']
acc = data['train accuracy']

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 15,
        }

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(epoch, loss, color='r')
ax1.set_ylabel('train loss', font)

ax2 = ax1.twinx()
ax2.plot(epoch, acc, color='g')
ax2.set_ylabel('train accuracy', font)
ax2.set_xlabel('epoch', font)

ax2.set_title('MNIST手写数字识别-全连接神经网络')
plt.xlabel('epoch', font)
plt.show()
