# --------------------------------------------------------------------------------------------------------
# 2019/04/18
# 0_ml_project_template - model.py
# md
# --------------------------------------------------------------------------------------------------------
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision as thv
from sklearn.decomposition import PCA, IncrementalPCA


class ResnetPlus(nn.Module):
    def __init__(self):
        super(ResnetPlus, self).__init__()
        self.resnet = thv.models.resnet152(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1024)
        # categorical features (country(3), place(7), verified(2), 19*labels(5))
        #       Jerimy Howard: embedding dimention =  (#categories+1)/2
        emb_dims = [(3, 2), (7, 4), (2, 2)] + [(6, 3)] * 19
        self.embeddings = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.emb_dropout = nn.Dropout(0.5)
        # continuous features (complexity, area, 19*distances)
        self.con_bn = nn.BatchNorm1d(21)
        self.meta_fc1 = nn.Sequential(nn.Linear(86, 128), nn.ReLU(), nn.Dropout(.1))
        self.meta_fc2 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(.2))
        self.meta_fc3 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.5))

        self.fc1 = nn.Sequential(nn.Linear(1088, 1024), nn.ReLU(), nn.Dropout(.1))
        self.fc2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(.3))
        self.fc3 = nn.Linear(1024, 6) # 5 categories + 0 for nan Todo: why is this here? Cleanup the code

    def forward(self, image, x_con, x_cat):
        # print(x_con.shape,x_cat.shape)
        # x_cat=x_cat.long()
        # print(x_cat)

        x_img = self.resnet(image)
        x_cat = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_cat = th.cat(x_cat, dim=1)
        x_cat = self.emb_dropout(x_cat)

        x_con = self.con_bn(x_con)
        x_meta = th.cat([x_cat, x_con], dim=1)
        x_meta = self.meta_fc1(x_meta)
        x_meta = self.meta_fc2(x_meta)
        x_meta = self.meta_fc3(x_meta)

        x = th.cat([x_meta, x_img], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class Resnet:
    # class Resnet(thv.models.resnet.ResNet):  # see https://discuss.pytorch.org/t/modify-resnet-or-vgg-for-single-channel-grayscale/22762/2
    def __init__(self, resnet_version=18, pretrained=True, n_classes=1000, fc_size=1024, dropout=.5):
        # super(Resnet, self).__init__()
        if resnet_version == 18:
            self.resnet = thv.models.resnet18
        elif resnet_version == 34:
            self.resnet = thv.models.resnet34
        elif resnet_version == 50:
            self.resnet = thv.models.resnet50
        elif resnet_version == 101:
            self.resnet = thv.models.resnet101
        elif resnet_version == 152:
            self.resnet = thv.models.resnet152
        else:
            print(f"Resnet version {resnet_version} doesn't exisit!")
            self.resnet = None
        self.net = self.resnet(pretrained=pretrained)
        # Replace the fc-layer
        self.net.fc = nn.Sequential(nn.Linear(self.net.fc.in_features, fc_size),
                                    nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(fc_size, n_classes))

    def unfreeze(self, unfreeze_layers=(1, 2, 3, 4)):
        unfreeze_layers_lst = []
        if 1 in unfreeze_layers: unfreeze_layers_lst.append(self.net.layer1)
        if 2 in unfreeze_layers: unfreeze_layers_lst.append(self.net.layer2)
        if 3 in unfreeze_layers: unfreeze_layers_lst.append(self.net.layer3)
        if 4 in unfreeze_layers: unfreeze_layers_lst.append(self.net.layer4)
        for layer in unfreeze_layers_lst:
            for param in layer.parameters():
                param.requires_grad = True
        return self.net

    @staticmethod
    def load(self, model_path):
        self.net = th.load(model_path)
        return self.net

    def features(self, x, from_layer=8):
        net_features = nn.Sequential(*list(self.net.children())[:4 + from_layer])
        return net_features(x)

    def logits(self, x):
        pass

    def __call__(self, x):
        self.net.forward(x)
        return self.net


class SimpleCNN(nn.Module):
    # From https://learning.oreilly.com/library/view/programming-pytorch-for/9781492045342/ch03.html
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = th.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    pass
