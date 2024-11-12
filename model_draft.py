import torch.nn as nn
import torch
import torch.nn.functional as F

#Need to add batching too
class CortexCodecBasicUModel(nn.Module):
    def __init__(self,steps=3, input_size = (32, 21*512), maxdepth=256, ourkernel=4, num_classes=19):
        super(CortexCodecBasicUModel).__init__()
        self.num_channels = input_size[0]
        self.num_samples = input_size[1]
        self.steps = steps
        self.ourkernel = ourkernel
        self.num_classes = num_classes
        self.starting = nn.Conv2d(in_channels=1,out_channels=maxdepth,kernel_size=(self.num_channels,1))
        self.ending = nn.Conv2d(maxdepth, self.num_samples) #need to fix that when i figure out the issue below
        self.stepsdown = []
        self.stepsup = []
        for i in range(steps):
            self.stepsdown.append(nn.Conv1d(maxdepth//(2**i),maxdepth//(2**(i+1)),kernel_size=ourkernel))
            self.stepsup.append(nn.Conv1d(maxdepth//(2**(steps-i)),maxdepth//(2**(steps-i-1)))) #Figure out how kernel can be set to make image larger without padding. Maybe Conv Transpose?
        self.relu = nn.ReLU()
        self.classifier = nn.Linear((self.num_samples*maxdepth)//((2*ourkernel)**steps),self.num_classes)
    def forward(self, data):
        x = self.starting(data)
        for i in range(self.steps):
            x = self.stepsdown[i](x)
            x = self.relu(x)
        tmp = x.flatten()
        tmp = self.classifier(tmp)
        for i in range(self.steps):
            x = self.stepsup[i](x)
            x = self.relu(x)
        return x, tmp #Loss function for x is matching the reconstruction EEG (maybe use autoencoder isntead then), tmp is for classifier. Need to add one for constructing music.