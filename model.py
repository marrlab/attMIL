import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        self.ClassCount = 5

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            # nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),

        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(32 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, self.ClassCount),
            nn.Softmax(1)
        )

        self.sil_classifier = nn.Sequential(
            nn.Linear(self.L * 1, 32),
            nn.Linear(32, self.ClassCount),
            nn.Softmax(1)
        )

    def forward_MILSILatt(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 32 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        Y_sil_prob = self.sil_classifier(H)

        return Y_prob, Y_hat, A, Y_sil_prob

    def forward_MIL_maxpooling(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 32 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        # H = F.sigmoid(H)
        M, _ = H.max(dim = 0)
        M = torch.unsqueeze(M, 0)
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()


        return Y_prob, Y_hat

    def forward_MIL_maxpooling_SIL(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 32 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        # H = F.sigmoid(H)
        M, _ = H.max(dim = 0)
        M = torch.unsqueeze(M, 0)
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        Y_sil_prob = self.sil_classifier(H)

        return Y_prob, Y_hat, Y_sil_prob



    def forward_SIL(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 32 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        Y_sil_prob = self.sil_classifier(H)

        return Y_sil_prob



    def c_o_MIL_maxpooling_SIL(self, X, target, sil_target,sil_ratio):
        target = target.float()
        sil_target = sil_target.float()

        Y_prob, _, silY_prob = self.forward_MIL_maxpooling_SIL(X)

        loss = nn.BCELoss()
        mil_bce_loss = loss(Y_prob, target)

        sil_loss = nn.BCELoss()
        sil_bce_loss = sil_loss(silY_prob, sil_target)

        loss = (1 - sil_ratio) * mil_bce_loss + sil_ratio * sil_bce_loss

        return mil_bce_loss, sil_bce_loss, loss



    def c_o_MIL_maxpooling(self, X, target):
        target = target.float()

        Y_prob , _ = self.forward_MIL_maxpooling(X)

        loss = nn.BCELoss()
        mil_bce_loss = loss(Y_prob,target)

        return mil_bce_loss


    def c_o_MILSILatt(self, X, target, sil_target, sil_ratio = 0):
        target = target.float()
        sil_target = sil_target.float()

        Y_prob, _, A, silY_prob = self.forward_MILSILatt(X)

        loss = nn.BCELoss()
        mil_bce_loss = loss(Y_prob,target)

        sil_loss = nn.BCELoss()
        sil_bce_loss = sil_loss(silY_prob, sil_target)


        loss = (1 - sil_ratio) * mil_bce_loss + sil_ratio * sil_bce_loss

        return mil_bce_loss, A, sil_bce_loss,loss







    def c_o_SIL(self, X, target):
        target = target.float()

        silY_prob = self.forward_SIL(X)
        sil_loss = nn.BCELoss()
        sil_bce_loss = sil_loss(silY_prob, target)

        return sil_bce_loss




    def save(self, modelname, modelDir="Model"):
        filename = modelname + "-" + time.strftime("%Y%m%d-%H%M%S") + ".mdl"
        path = os.path.join(modelDir,filename)
        torch.save(self, path)

    def load_latest(self, modelDir="Model", modelname =[]):
        if modelname == []:
            model_name = sorted(os.listdir(modelDir), reverse=True)[0]
        else:
            model_name = modelname

        return torch.load(os.path.join(modelDir, model_name),map_location=torch.device('cpu'))
