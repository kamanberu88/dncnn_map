import argparse
import os
import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import DnCNN
import matplotlib.pyplot as plt
from utils import AverageMeter
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model=DnCNN(num_layers=17)
model = model.to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters())

from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset
import torch
import os
import numpy as np
import random

class NoisyCustomDataset(Dataset):
    def __init__(self, img_dir, transform=None, noise_factor=0.5, salt_pepper_ratio=0.05):
        self.img_dir = img_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform
        self.noise_factor = noise_factor
        self.salt_pepper_ratio = salt_pepper_ratio

    def add_salt_pepper_noise(self, image):
        image = np.array(image)  # Convert PIL Image to numpy array
        salt_ratio = self.salt_pepper_ratio
        pepper_ratio = 1 - self.salt_pepper_ratio

        num_salt = np.ceil(salt_ratio * image.size * self.noise_factor)
        num_pepper = np.ceil(pepper_ratio * image.size * self.noise_factor)

        # Add Salt noise
        salt_coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
        image[salt_coords[0], salt_coords[1]] = 255

        # Add Pepper noise
        pepper_coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
        image[pepper_coords[0], pepper_coords[1]] = 0

        return Image.fromarray(image.astype(np.uint8))  # Convert numpy array back to PIL Image

    def add_brightness_contrast_change(self, image):
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.5, 1.5))  # ランダムに明度を変更
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.5, 1.5))  # ランダムにコントラストを変更
        return image

    def add_blur(self, image):
        return image.filter(ImageFilter.GaussianBlur(radius=random.randint(0, 3)))  # ランダムにぼかしを追加

    def add_radial_noise(self, image):
        image = np.array(image)
        x, y = image.shape[0], image.shape[1]
        center_x, center_y = np.random.randint(0, x), np.random.randint(0, y)
        strength = np.random.uniform(0, 255)
        for i in range(x):
            for j in range(y):
                distance = np.sqrt((center_x - i) ** 2 + (center_y - j) ** 2)
                if distance == 0:
                    distance = 1  # 0除算を防ぐために距離が0のときは1を代入します
                change = np.random.uniform(0, 1) * strength / distance
                pixel = image[i, j] + change
                pixel = np.clip(pixel, 0, 255)
                image[i, j] = pixel
        return Image.fromarray(image.astype(np.uint8))

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_names[index])
        image = Image.open(img_path).convert('L')

        # ランダムにノイズのタイプを選びます
        noise_type = random.choice(['gaussian', 'salt_pepper', 'brightness_contrast', 'blur', 'radial'])

        if noise_type == 'gaussian':
            image_np = np.array(image)
            noisy_image_np = image_np + self.noise_factor * np.random.randn(*image_np.shape)
            noisy_image_np = np.clip(noisy_image_np, 0, 255)
            noisy_image = Image.fromarray(noisy_image_np.astype(np.uint8))
        elif noise_type == 'salt_pepper':
            noisy_image = self.add_salt_pepper_noise(image)
        elif noise_type == 'brightness_contrast':
            noisy_image = self.add_brightness_contrast_change(image)
        elif noise_type == 'blur':
            noisy_image = self.add_blur(image)
        elif noise_type == 'radial':
            noisy_image = self.add_radial_noise(image)
        
        if self.transform:
            noisy_image = self.transform(noisy_image)
            image = self.transform(image)

        return noisy_image, image

    def __len__(self):
        return len(self.img_names)
    
transform = transforms.Compose([
      transforms.Grayscale(),
     
    transforms.ToTensor(),
])

# 自作のデータセットを作成
dataset = NoisyCustomDataset(img_dir='C:\\Users\\kaman\\Desktop\\jpg8k_me\\400', transform=transform)
batch_size=2

dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                           )

epochs=30


history={'loss':[]}

for epoch in range(epochs):
        #epoch_losses = AverageMeter()
    train_loss=0
        #with tqdm(total=(len(dataset) - len(dataset) % batch_size)) as _tqdm:
            #tqdm.set_description('epoch: {}/{}'.format(epoch + 1, epoch))
    for data in tqdm(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs)

            loss = criterion(preds, labels) / (2 * len(inputs))
            train_loss+=loss.item()

            #epoch_losses.update(loss.item(), len(inputs))

            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_train_loss = train_loss / len(dataloader)

    history['loss'].append(avg_train_loss)



    if (epoch + 1) % 1 == 0:
                 print("epoch{} train_loss:{:.4} ".format(epoch,avg_train_loss))

PATH = './reeeeusult3_net.pth'
torch.save(model.state_dict(), PATH)

plt.plot(history['loss'],
         marker='.',
         label='loss(Training)')


plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()



        #torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format(opt.arch, epoch)))


