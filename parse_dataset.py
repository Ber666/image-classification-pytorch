import struct
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def parse(path):
    with open(path, "rb") as tmp:
        magic = tmp.read(4)
        '''
        datatype, ndim = magic[2:4]
        print(datatype, ndim)
        '''
        ndim = magic[3]
        pat = '>'
        pat += ndim * 'I'
        dims = struct.unpack(pat, tmp.read(4 * ndim))
        npdata = np.fromfile(tmp, dtype=np.uint8)
        return dims, npdata


def imshow(img, tag, h, w, i=0):
    print("supposed to be %d" % tag[i])
    img = img / 2 + 0.5
    npimg = img[i * w * h: (i + 1) * w * h].reshape(h, w)
    plt.imshow(npimg)
    plt.show()


def main():
    dims, npdata = parse("dataset/train-images.idx3-ubyte")
    datasize, height, width = dims
    print(datasize, height, width)
    print(npdata.shape)

    dimtag, nptag = parse("dataset/train-labels.idx1-ubyte")
    print(dimtag)
    imshow(npdata, nptag, height, width, 2)

    # bug1: used a zipped file
    # bug2: np reshape cannot be multidimensional


if __name__ == '__main__':
    main()

