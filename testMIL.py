import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from model import Attention
from DataLoader import DataLoader

import sys
import os
import pickle

Classes = []
with open("data-features/classes.txt") as f:
    clsdata = f.readlines()
    for cls in clsdata:
        Classes.append(cls.strip("\n"))


cuda = torch.cuda.is_available()

torch.manual_seed(1)

if cuda:
    torch.cuda.manual_seed(1)

loader_kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


# Load the model
trainwhat = sys.argv[1]
samplewise = sys.argv[2]
imgmode = sys.argv[3]

all_models = [x for x in os.listdir("Model") if not x.startswith(".") and x.split("-")[1] == trainwhat and x.split("-")[2] == samplewise]


models = []

for run in range(1,6):
    m = []
    for fold in range(3):
        run_models = [x for x in all_models if x.split("-")[0] == str(run) and x.split("-")[3] == str(fold)]
        run_models = sorted(run_models)
        if run_models == []:
            continue
        m.append(run_models[-1])
    models.append(m)




for fold in range(3):
    test_loader = data_utils.DataLoader(DataLoader(foldnumber=fold,
                                                   mode=samplewise,
                                                   imgmode=imgmode,
                                                   train=False),
                                        batch_size=1,
                                        shuffle=False,
                                        **loader_kwargs)
    model_runs = []
    print("Analyzing fold : " + str(fold))
    for run in range (1,6):
        print("Analyzing run : " + str(run))
        model_name = models[run-1][fold]
        model = torch.load(os.path.join("Model", model_name), map_location=torch.device('cpu'))

        if cuda:
            model.cuda()

        res = []
        gt = []
        model.eval()


        for batch_idx, (data, label) in enumerate(test_loader):
            if cuda:
                data = data.cuda()
                label = label.cuda()

            data, label = Variable(data), Variable(label)
            if data.nelement() == 0:
                continue

            if trainwhat == "SIL":
                out = model.forward_SIL(data)
            elif trainwhat == "MILSILatt":
                out = model.forward_MILSILatt(data)
            elif trainwhat == "MILmax":
                out = model.forward_MIL_maxpooling(data);
            elif trainwhat == "MILSILmax":
                out = model.forward_MIL_maxpooling_SIL(data)


            result = out[0].data.cpu().numpy()
            res.append(result)
            gt.append(label.cpu().data.int().numpy()[0])
        model_runs.append([res, gt])



    with open("evals/" + trainwhat + "-" + samplewise + "-" + imgmode + "-test-" + str(fold) + ".pkl", "wb") as f:
        pickle.dump(model_runs, f)

    print("done with fold")

print("job finished")