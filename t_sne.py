from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
import torch

 
head_ckpt = torch.load('head101.pth')
train_data = head_ckpt['kernel'].data.cpu()
train_data = np.array(train_data)
X = train_data
race = np.concatenate([np.ones(1543),np.ones(1728)*2,np.ones(11326)*3,np.ones(1923)*4])
color=race
n_neighbors = 10
n_components = 3
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
# Y = tsne.fit_transform(X)  # 转换后的输出
# np.save('stne3.npy',Y)
# np.save('race.npy',color)
Y = np.load('stne3.npy')
fig = plt.figure(figsize=(8, 8))


ax = Axes3D(fig)
ax.scatter(Y[3271:14597, 0], Y[3271:14597, 1],Y[3271:14597, 2], c='b', cmap=plt.cm.Spectral,label='caucasian')
ax.scatter(Y[:1543, 0], Y[:1543, 1],Y[:1543, 2], c='r', cmap=plt.cm.Spectral,label='african')
ax.scatter(Y[1543:3271, 0], Y[1543:3271, 1], Y[1543:3271, 2],c='g', cmap=plt.cm.Spectral,label='asian')
ax.scatter(Y[14597:, 0], Y[14597:, 1],  Y[14597:, 2],c='y', cmap=plt.cm.Spectral,label='indian')
plt.legend(loc="upper left")
plt.show()
