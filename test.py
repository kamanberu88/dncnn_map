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
import numpy as np
import PIL.Image as pil_image
cudnn.benchmark = True
import PIL
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from model import DnCNN
model_path = 'reusult2_net.pth'
model=DnCNN(num_layers=17)
model=model.to(device)
model.load_state_dict(torch.load(model_path))

model.eval()
#input = input.resize((original_width, original_height), resample=pil_image.BICUBIC)
#descriptions += '_sr_s{}'.format(opt.downsampling_factor)
#input.save(os.path.join(opt.outputs_dir, '{}{}.png'.format(filename, descriptions)))
input = np.array(input).astype(np.float32)
input /= 255.0
input_path="C:\\Users\\kaman\\Desktop\\jpg8k_me\\407\\sim0003.jpg"
with torch.no_grad():
        input = transforms.ToTensor()(input).unsqueeze(0).to(device)
        pred = model(input)
        output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        output = pil_image.fromarray(output, mode='RGB')

output.show()
