import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from model import Attention
from DataLoader import DataLoader

import sys

init_with = "new"


cuda = torch.cuda.is_available()

torch.manual_seed(1)

if cuda:
    torch.cuda.manual_seed(1)

loader_kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


foldno = int(sys.argv[1])
trainwhat = sys.argv[2]
samplewise = sys.argv[3]
imgmode = sys.argv[4]
run_number = sys.argv[5]

# Set dataloader params for the selected model
if trainwhat == "SIL":
    cb = DataLoader(foldno,
                    imgmode="bag",
                    mode=samplewise)
elif trainwhat == "MILSILatt":
    cb = DataLoader(foldno,
                    imgmode=imgmode,
                    mode=samplewise)
elif trainwhat == "MILmax":
    cb = DataLoader(foldno,
                    imgmode=imgmode,
                    mode=samplewise)
elif trainwhat == "MILSILmax":
    cb = DataLoader(foldno,
                    imgmode=imgmode,
                    mode=samplewise)

train_loader = data_utils.DataLoader(cb,
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)


# initialize the weights, either randomly or based on the last trained model run
if init_with == "new":
    model = Attention()
elif init_with == "latest":
    model = Attention().load_latest()


if cuda:
    model.cuda()

# use the same adam optimizer for all models:
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=10e-5, amsgrad=True)



def trainMILSILmax(epoch):
    model.train()
    train_loss = 0.
    train_sil_loss = 0.
    train_mil_loss = 0.

    # calculate the coefficient at each epoch to weight the SIL loss
    sil_ratio = torch.tensor(1 * (1 - .5) ** epoch)


    for batch_idx, (data, label) in enumerate(train_loader):
        # sil target is just replicates of the weak bag labels
        sil_target = label.repeat((data.shape[1], 1))

        if cuda:
            data = data.cuda()
            label = label.cuda()
            sil_target = sil_target.cuda()
            sil_ratio = sil_ratio.cuda()

        data, label, sil_target, sil_ratio = Variable(data), Variable(label), Variable(sil_target), Variable(sil_ratio)

        # if bag data is not extracted/saved/loaded successfully just ignore this bag
        if data.nelement() == 0:
            continue

        optimizer.zero_grad()

        # forward the model
        mil_bce_loss, sil_bce_loss, loss = model.c_o_MIL_maxpooling_SIL(data, label, sil_target, sil_ratio)
        train_loss += loss.data
        train_mil_loss += mil_bce_loss.data
        train_sil_loss += sil_bce_loss.data

        # backward pass
        loss.backward()

        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_sil_loss /= len(train_loader)
    train_mil_loss /= len(train_loader)

    # print everything
    print('Epoch: {}, Loss: {:.4f}, MIL Loss: {:.4f}, SIL Loss: {:.4f}, SIL Ratio: {:.4f}'.format(epoch,
                                                                                                              train_loss.cpu().numpy(),
                                                                                                              train_mil_loss.cpu().numpy(),
                                                                                                              train_sil_loss.cpu().numpy(),
                                                                                                              sil_ratio.cpu().numpy()
                                                                                                              )
          )

    return train_mil_loss


def trainMILmax(epoch):
    model.train()
    train_loss = 0.
    train_mil_loss = 0.

    # train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):


        if cuda:
            data = data.cuda()
            label = label.cuda()


        data, label = Variable(data), Variable(label)

        if data.nelement() == 0:
            continue


        optimizer.zero_grad()

        loss = model.c_o_MIL_maxpooling(data, label)
        train_loss += loss.data




        # backward pass
        loss.backward()

        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)


    print('Epoch: {}, Loss: {:.4f}'.format(epoch,
                                               train_loss.cpu().numpy(),
                                           )
    )




    return train_mil_loss



def trainMILSILatt(epoch):
    model.train()
    train_loss = 0.
    train_sil_loss = 0.
    train_mil_loss = 0.

    sil_ratio = torch.tensor(1 * (1 - .05) ** epoch)


    # train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):

        sil_target = label.repeat((data.shape[1], 1))

        if cuda:
            data = data.cuda()
            label = label.cuda()
            sil_target = sil_target.cuda()
            sil_ratio = sil_ratio.cuda()

        data, label, sil_target, sil_ratio = Variable(data), Variable(label), Variable(sil_target), Variable(sil_ratio)

        if data.nelement() == 0:
            continue


        optimizer.zero_grad()

        mil_bce_loss,_ , sil_bce_loss, loss = model.c_o_MILSILatt(data, label, sil_target, sil_ratio)
        train_loss += loss.data
        train_mil_loss += mil_bce_loss.data
        train_sil_loss += sil_bce_loss.data


        # backward pass
        loss.backward()

        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_sil_loss /= len(train_loader)
    train_mil_loss /= len(train_loader)

    for param_group in optimizer.param_groups:
        lr = param_group["lr"]

    print('Epoch: {}, Loss: {:.4f}, MIL Loss: {:.4f}, SIL Loss: {:.4f}, SIL Ratio: {:.4f}, lr: {:.4f}'.format(epoch,
                                                                                               train_loss.cpu().numpy(),
                                                                                               train_mil_loss.cpu().numpy(),
                                                                                               train_sil_loss.cpu().numpy(),
                                                                                               sil_ratio.cpu().numpy(),
                                                                                                lr)
          )




    return train_mil_loss



def trainSIL(epoch):
    model.train()
    train_sil_loss = 0.

    for batch_idx, (data, label) in enumerate(train_loader):

        sil_target = label.repeat((data.shape[1], 1))

        if cuda:
            data = data.cuda()
            sil_target = sil_target.cuda()


        data, sil_target = Variable(data), Variable(sil_target)

        optimizer.zero_grad()

        loss = model.c_o_SIL(data, sil_target)
        train_sil_loss += loss.data

        # backward pass
        loss.backward()

        # step
        optimizer.step()

        # calculate loss and error for epoch

    train_sil_loss /= len(train_loader)



    print('Epoch: {}, SIL Loss: {:.4f}'.format(epoch,train_sil_loss.cpu().numpy()))


    return train_sil_loss




if __name__ == "__main__":
    print('Start Training')
    epoch = 0
    end = 0
    while epoch < 150:
        if trainwhat == "SIL":
            loss = trainSIL(epoch)
        elif trainwhat == "MILSILatt":
            loss = trainMILSILatt(epoch)
        elif trainwhat == "MILmax":
            loss = trainMILmax(epoch)
        elif trainwhat == "MILSILmax":
            loss = trainMILSILmax(epoch)

        if loss < 0.001:
            end -=- 1
        if end == 7:
            print("training finished, low loss reached.")
            break
        if epoch % 10 == 0:
            model.save(run_number + "-" + trainwhat + "-" + samplewise + "-" + str(foldno))

        epoch -=- 1
    model.save(run_number + "-" + trainwhat + "-" + samplewise + "-" + str(foldno))
