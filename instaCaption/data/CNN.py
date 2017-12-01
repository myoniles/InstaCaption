import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.trainer as trainer
import torch.utils.trainer.plugins
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn


cudnn.benchmark = True

model = models.resnet152(pretrained=True)
model = torch.nn.DataParallel(model).cuda()

print(model)


normalize= transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trainData = datasets.ImageFolder(
        "./resized",
        transforms.Compose([transforms.RandomSizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))

train_loader = torch.utils.data.DataLoader(trainData)

# define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), 0.01 , 0.9)


model.train()
for i, (input, target) in enumerate(train_loader):
    target = target.cuda(async=True)
    input_var = Variable(input)
    target_var = Variable(target)

    output=model(input_var)
    loss = criterion(output, target_var)

    print(output)
    print(loss)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




"""
# pass model, loss, optimizer and dataset to the trainer
t = trainer.Trainer(model, criterion, optimizer, train_loader)

# register some monitoring plugins
t.register_plugin(trainer.plugins.ProgressMonitor())
t.register_plugin(trainer.plugins.AccuracyMonitor())
t.register_plugin(trainer.plugins.LossMonitor())
t.register_plugin(trainer.plugins.TimeMonitor())
t.register_plugin(trainer.plugins.Logger(['progress', 'accuracy', 'loss', 'time']))

# train!
t.run(90)
"""
