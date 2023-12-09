import torch
from torch import nn
import torch.optim as optim
import sys
# import graphviz
from pyvis.network import Network


class SimpleLayer(nn.Module):
    def __init__(self,channel_in,channel_hid,channel_out):
        super(SimpleLayer, self).__init__()
        self.conv_1 = nn.Conv2d(channel_in, channel_hid ,3,1,0, bias=False)  # 输入1,4,4 输出 1,2,2
        self.conv_2 = nn.Conv2d(channel_hid, channel_out ,2,1,0, bias=False)  # 输入1,2,2 输出 1,1,1
        # self.fc_2 = nn.Linear(channel_out, channel_out, bias=False)
        self.init_weights()

    def init_weights(self):
        # 使用torch.arange生成整数张量
        self.conv_1.weight.data = (torch.arange(9)/10).float().view(self.conv_1.weight.shape)
        self.conv_2.weight.data = (torch.arange(4)/10).float().view(self.conv_2.weight.shape)

    def forward(self, x0):
        x1 = self.conv_1(x0.float())
        y1 = torch.sigmoid(x1)
        x2 = self.conv_2(y1)
        y2 = torch.sigmoid(x2)
        return [x1,y1,x2,y2]


if __name__ == '__main__':
    input = torch.reshape(torch.arange(1,17), (1, 1, 4, 4))/10
    model = SimpleLayer(1,1,1)
    target = torch.tensor([1]).view(1,1,1,-1).to(torch.float32)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    manual_vec_1 = 0
    manual_vec_2 = 0
    # sys.exit()

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output[-1], target)

        print('权重（更新前）：',model.conv_1.weight.data)
        print('权重（更新前）：',model.conv_2.weight.data)

        loss.backward()
        gard_1 = model.conv_1.weight.grad
        gard_2 = model.conv_2.weight.grad

        manual_vec_1 = gard_1 + 0.9 * manual_vec_1
        manual_vec_2 = gard_2 + 0.9 * manual_vec_2

        manual_nest_1 = gard_1 + 0.9 * manual_vec_1
        manual_nest_2 = gard_2 + 0.9 * manual_vec_2

        manual_data_1 = model.conv_1.weight.data  - 0.01  * manual_nest_1
        manual_data_2 = model.conv_2.weight.data  - 0.01  * manual_nest_2

        print(f' -{epoch}','---' * 19)

        # print('速度w1',manual_vec_1,'\n速度w2',manual_vec_2)
        print('权重（手动）w1：',manual_data_1,'\n权重（手动）w2：',manual_data_2)
        optimizer.step()
        print('权重（更新后）w1：',model.conv_1.weight.data)
        print('权重（更新后）w2：',model.conv_2.weight.data)

        print('==='*20,'\n')
        # break

