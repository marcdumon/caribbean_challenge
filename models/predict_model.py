import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def predict_image(model: str, transforms, img, device: str = 'cuda', labels: bool = True) -> np.array:
    predict_model = th.load(model).to(device)
    predict_model.eval()
    img_tensor = transforms(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    y = predict_model(img_tensor)
    y = F.softmax(y, dim=1)
    if labels: y = th.argmax(y, dim=1)
    y = y.cpu().detach().numpy()
    return y[0]


if __name__ == '__main__':
    fp = '/media/md/Development/My_Projects/drivendata_open_ai_caribbean_challenge/data/processed/'
    valid = pd.read_csv(fp + 'valid.csv')
    others = valid.loc[valid['label'] == 'other']
    x = others.iloc[0]
    img_id = x['id']
    img_label = x['label']
    img = Image.open(fp + f'train_valid_test_augmented/{img_id}.png').convert('RGB')

    model = '../../models/Resnet152_Lin1024_Drop50_Size512_Normal-best_model_4_0.537269961114826.pth.tar'
    transforms = transforms.Compose([transforms.Resize(512),
                                     transforms.CenterCrop(512),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])
    labels = ['healthy_metal', 'irregular_metal', 'concrete_cement', 'incomplete', 'other']
    print(img_label, predict_image(model, transforms, img, labels=False))
