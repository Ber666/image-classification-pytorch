import parse_dataset
import my_nn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

BATCH_SIZE = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
testx_dims, np_testx = parse_dataset.parse('dataset/t10k-images.idx3-ubyte')
testy_dims, np_testy = parse_dataset.parse('dataset/t10k-labels.idx1-ubyte')
testx = torch.from_numpy(np_testx).view(testx_dims[0], 1, testx_dims[1], testx_dims[2]).float()
testx = (testx - 128)/128
testy = torch.from_numpy(np_testy).long()

test_dataset = Data.TensorDataset(testx, testy)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
net = my_nn.Net()
resume_file = 'checkpoint.pth'
print("=> loading checkpoint '{}'".format(resume_file))
checkpoint = torch.load(resume_file)
net.load_state_dict(checkpoint['state_dict'])
net.to(device)
net.eval()  # change to test mode
rightanswer = 0.0
for step, (batch_x, batch_y) in enumerate(test_loader):
    inputs = batch_x.to(device)
    target = batch_y.to(device)
    rightanswer += (net(inputs).argmax(axis=1) == target).sum()
    # print(rightanswer)
print('%d out of %d answers are right'%(rightanswer, testx_dims[0]))