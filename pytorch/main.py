import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import unet3d
import config
from loader import NIIloader
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# prepare your own data
path = '/Users/jinxiaoqiang/jinxiaoqiang/数据集/Bone/ribfrac/ribfrac-val-images'
transform =transforms.Compose([transforms.ToTensor()])
train_set=NIIloader(path,dataset='train',transform=transform)
train_loader = DataLoader(train_set, batch_size = config.batch_size, shuffle=True)

# prepare the 3D model
class TargetNet(nn.Module):
    def __init__(self, base_model, n_class=2):
        super(TargetNet, self).__init__()

        self.base_model = base_model
        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, n_class, bias=True)

    def forward(self, x):
        self.base_model(x)
        self.base_out = self.base_model.out512
        self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(
            self.base_out.size()[0], -1)
        self.linear_out = self.dense_1(self.out_glb_avg_pool)
        final_out = self.dense_2(F.relu(self.linear_out))
        return final_out


base_model = unet3d.UNet3D()
# Load pre-trained weights
weight_dir = 'pretrained_weights/Genesis_Chest_CT.pt'
checkpoint = torch.load(weight_dir,map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
base_model.load_state_dict(unParalled_state_dict)
target_model = TargetNet(base_model)
target_model.to(config.device)
target_model = nn.DataParallel(target_model, device_ids=[i for i in range(torch.cuda.device_count())])
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(target_model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)

# train the model

for epoch in range(config.intial_epoch, config.nb_epoch):
    target_model.train()
    for batch_ndx, (x, y) in enumerate(train_loader):
        x, y = x.float().to(config.device), y.float().to(config.device)
        pred = F.sigmoid(target_model(x))
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss:{}".format(loss))