from .model import ResNet18
from .feeds import FMNIST # Fashion MNIST dataset

EPOCH = 1000
BATCH = 256
RATIO = 0.5 # part of samples of a batch are positive, others are negative.


if __name__ == '__main__':
    net = ResNet18()
    data = FMNIST('~/sr/data/FashionMNIST')
    total = len(data[0])
    H, W = data[0][0].shape[:2]
    CLS = 10

    for i in range(EPOCH):
        p = np.random.permutation(np.arange(total))
        n = total // BATCH
        for j in range(n):
            # generate a batch
            
            bx = data[0][p[i*BATCH:(i+1)*BATCH]]
            by = data[1][p[i*BATCH:(i+1)*BATCH]]
            # convert image from NHWC into NCHW
            x = bx.reshape([BATCH, 1, H, W]) 
            # convert integer labels into hot vectors
            y = np.eye(CLS)[by]



            

