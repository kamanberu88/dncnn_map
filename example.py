import argparse
import os
import io
import numpy as np
import PIL.Image as pil_image
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
from model import DnCNN
import PIL
from PIL import Image
import matplotlib.pyplot as plt
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
outputs_dir="./outputs_dir"
if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

model_path = './reeeeusult3_net.pth'
model=DnCNN(num_layers=17)
model=model.to(device)
state_dict =model.load_state_dict(torch.load(model_path))
"""
for n, p in torch.load(weights_path, map_location=lambda storage, loc: storage).items():
    if n in state_dict.keys():
        state_dict[n].copy_(p)
    else:
        raise KeyError(n)
"""

model=DnCNN(num_layers=17)
model=model.to(device)
model.eval()
#C:\Users\kaman\Desktop\jpg8k_me\tetst
image_path="C:\\Users\\kaman\\Desktop\\jpg8k_me\\tetst\\sim0004.jpg"

filename = os.path.basename(image_path).split('.')[0]
descriptions = ''


#input = pil_image.open(image_path).convert('R')
#input_open=PIL.Image.open(image_path)
#input_open.show()
input=Image.open(image_path)
#print (input.mode)
input.show()
transformS=transforms.Compose([
        transforms.Grayscale(),
       transforms.ToTensor()
])
#input = transforms.ToTensor()(input).unsqueeze(0).to(device)
input = transformS(input).unsqueeze(0).to(device)
#print(input.size())
print("Input size:", input.size())


with torch.no_grad():
    pred = model(input)
     #print(pred)
    print("Output size:", pred.size())


"""
output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).byte().cpu().numpy()
output = output.transpose((1, 2, 0))  

output = pil_image.fromarray(output)
plt.imshow(output)
plt.axis('off')  # Turn off axes
plt.show()
#output.save(os.path.join(outputs_dir, '{}_{}.jpg'.format(filename, descriptions)))
"""
output = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).byte().cpu().numpy()
output = np.squeeze(output, axis=0)  # Remove the first axis with size 1
output = output.astype(np.uint8)  # Convert the data type to uint8

output = pil_image.fromarray(output)

# Display the output image
plt.imshow(output, cmap='gray')
plt.axis('off')
plt.show()


