import argparse
import os
import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import DnCNN

from utils import AverageMeter
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model=DnCNN(num_layers=17)
model = model.to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters())
from PIL import Image
from torch.utils.data import Dataset
import torch
import os
import numpy as np

class NoisyCustomDataset(Dataset):
    def __init__(self, img_dir, transform=None, noise_factor=0.5):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform
        self.noise_factor = noise_factor

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index])
        
        # 画像をロードし、必要なら変換を適用
        image = Image.open(img_path).convert('L')  # グレースケールで読み込む場合
        if self.transform:
            image = self.transform(image)
        
        # 画像にランダムなガウスノイズを追加
        noisy_image = image + self.noise_factor * torch.randn(*image.shape)
        noisy_image = torch.clamp(noisy_image, 0., 1.)
        
        return noisy_image, image

    def __len__(self):
        return len(self.img_names)

# 変換を定義（必要に応じて調整）
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 自作のデータセットを作成
dataset = NoisyCustomDataset(img_dir='C:\\Users\\kaman\\Desktop\\jpg8k_me\\400', transform=transform)


dataloader = DataLoader(dataset=dataset,
                            batch_size=64,
                            shuffle=True,
                           )

epoch=100
batch_size=64
for epoch in range(epoch):
        epoch_losses = AverageMeter()

        with tqdm(total=(len(dataset) - len(dataset) % batch_size)) as _tqdm:
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, epoch))
            for data in dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels) / (2 * len(inputs))

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _tqdm.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                _tqdm.update(len(inputs))
                print(epoch_losses)

        #torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format(opt.arch, epoch)))
