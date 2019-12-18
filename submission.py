# --------------------------------------------------------------------------------------------------------
# 2019/12/18
# src - submission.py
# md
# --------------------------------------------------------------------------------------------------------
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms

from models.predict_model import predict_image

fp = '/media/md/Development/My_Projects/drivendata_open_ai_caribbean_challenge/data/processed/'
test = pd.read_csv(fp + 'test.csv')
model = '../models/best_models/resnet1_256_0_40_1e-05_20191217_23596-best_model_1_0.34451216677020136.pth.tar'
tfms = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
submission = pd.DataFrame(columns=['id', 'concrete_cement', 'healthy_metal', 'incomplete', 'irregular_metal', 'other'])
for i, row in test.iterrows():
    print(i, row['id'])
    img = Image.open(fp + f'train_valid_test_augmented/{row["id_aug"]}.png')

    y = predict_image(model, tfms, img, 'cuda', False)
    submission.loc[i] = [row['id']] + list(y)
submission.set_index('id', drop=True, inplace=True)
submission.to_csv('../models/submission_models/submission.csv')
print(submission)

if __name__ == '__main__':
    pass
