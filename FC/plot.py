import pandas as pd
import matplotlib.pyplot as plt

device_type = input("CPU or GPU\n")
csv_path = './{}/loss&acc.csv'.format(device_type)
data = pd.read_csv(csv_path)

epoch = data['epoch']
train_loss = data['train loss']
train_acc = data['train accuracy']
validate_loss = data['validate loss']
validate_acc = data['validate accuracy']

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 15,
        }

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(epoch, train_loss, color='r', label='training loss')
ax1.plot(epoch, validate_loss, color='b', label='validation loss')
ax1.legend()
ax1.set_title('Training and validation loss')

ax2 = fig.add_subplot(122)
ax2.plot(epoch, train_acc, color='r', label='training accuracy')
ax2.plot(epoch, validate_acc, color='b', label='validation accuracy')
ax2.legend()
ax2.set_title('Training and validation accuracy')

plt.show()
