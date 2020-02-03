import parse_dataset
import my_nn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

BATCH_SIZE = 16
N_EPOCH = 100
LR = 1e-4
FROM_START = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trainx_dims, np_trainx = parse_dataset.parse('dataset/train-images.idx3-ubyte')
trainy_dims, np_trainy = parse_dataset.parse('dataset/train-labels.idx1-ubyte')
trainx = torch.from_numpy(np_trainx).view(trainx_dims[0], 1, trainx_dims[1], trainx_dims[2]).float()
trainx = (trainx - 128)/128
trainy = torch.from_numpy(np_trainy).long()

'''
trainy = torch.zeros((trainy_dims[0], 10)).long()
for i, tag in enumerate(np_trainy):
    trainy[i][tag] = 1
'''
# print(trainy[0:10])
print(trainx.shape, trainy.shape)
net = my_nn.Net()
net.to(device)
start_epoch = 0
optimizer = optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

if FROM_START == False:
    try:
        resume_file = 'checkpoint.pth'
        print("=> loading checkpoint '{}'".format(resume_file))
        checkpoint = torch.load(resume_file)
        start_epoch = checkpoint['epoch']
        # best_score = checkpoint['best_score']
        net.load_state_dict(checkpoint['state_dict'])

    except FileNotFoundError:
        print("checkpoint not found, training from the start")


torch_dataset = Data.TensorDataset(trainx, trainy)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    # num_workers=2,              # subprocesses for loading data
)


for epoch in range(start_epoch, N_EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):
        inputs = batch_x.to(device)
        target = batch_y.to(device)
        # forward
        out = net(inputs)
        loss = criterion(out, target)
        # backward
        optimizer.zero_grad()  # clear all the gradient
        loss.backward()
        optimizer.step()

    print("loss:{}".format(loss.data))
    if epoch % 10 == 0:
        state = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 'best_score': best_score,
        }
        torch.save(state, 'checkpoint.pth')


# bug1: dim didn't match: the dim of data feeded in is: batchsize * channel * h * w
#       and the reminder confused me by showing me the shape of parameters
# bug2: didn't convese the data to float

# It then ran successfully, but when using GPU and change to cross-entropy...

# bug3: GPU out of memory -> use batch training
# https://blog.csdn.net/sjtuxx_lee/article/details/83112144
# two other bugs after changing to gpu
