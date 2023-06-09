import argparse
import os
import io
import numpy as np
import PIL.Image as pil_image
import torch
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
from torchvision import transforms
from model import DnCNN
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from models.model_rdnn import RRDN
from model import DnCNN
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_path = './rdnn_512_2.pth'
#model=DnCNN(17)
model=RRDN()
model=model.to(device)
state_dict =model.load_state_dict(torch.load(model_path))


model.eval()
#C:\Users\kaman\Desktop\jpg8k_me\tetst
image_path="/home/natori21_u/JAXA_database/512/jpg8k/407/sim0001.jpg"

filename = os.path.basename(image_path).split('.')[0]
descriptions = ''


#input = pil_image.open(image_path).convert('R')
#input_open=PIL.Image.open(image_path)
#input_open.show()
input=Image.open(image_path)
#print (input.mode)
#input.show()
transformS=transforms.Compose([
        transforms.Grayscale(),
        #transforms.RandomCrop(128),
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
    save_image(pred, './out/{}_{}.jpg'.format(filename, descriptions))
    #pred = pred.squeeze(0)  # Remove batch dimension
    #pred = pred.permute(1, 2, 0)  # Change (C, H, W) to (H, W, C)

    #img_numpy = pred.cpu().numpy()

    # Clip values to [0, 1] if they are floating point

    # Convert to proper numpy array for grayscale or color


    #plt.imshow(img_numpy)
    #plt.axis('off')  # Turn off axes
    #plt.show()
    #img_numpy = pred.cpu().numpy()
    #output = np.squeeze(img_numpy, axis=0)
    # convert to numpy array
    #img_numpy = np.transpose(img_numpy, (1, 2, 0))
    #print(img_numpy.size)
    #plt.imshow(img_numpy)
    #plt.axis('off')  # Turn off axes
    #plt.show()


"""
output = pred.byte().cpu().numpy()
output = output.transpose((1, 2, 0))  

output = pil_image.fromarray(output)
plt.imshow(output)
plt.axis('off')  # Turn off axes
plt.show()
#output.save(os.path.join(outputs_dir, '{}_{}.jpg'.format(filename, descriptions)))
"""



"""
# Display the output image
plt.imshow(output, cmap='gray')
plt.axis('off')
plt.show()
"""

