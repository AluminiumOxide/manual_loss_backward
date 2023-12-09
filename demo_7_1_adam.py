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
    # weight_decay=0,amsgrad=False 按照默认值,不是因为为了与原优化器匹配,而是,我懒的多写几行代码以及加几行判断,6_3又不是没有
    optimizer = optim.Adam(model.parameters(), lr=0.01,betas=(0.9, 0.999),eps=1e-08)

    manual_m_1 = 0
    manual_v_1 = 0
    manual_m_2 = 0
    manual_v_2 = 0
    BETA1 = 0.9
    BETA2 = 0.999

    for epoch in range(5):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output[-1], target)
        loss.backward()

        gard_1 = model.conv_1.weight.grad
        gard_2 = model.conv_2.weight.grad
        data_1 = model.conv_1.weight.data
        data_2 = model.conv_2.weight.data


        manual_m_1 = BETA1 * manual_m_1 + (1 - BETA1) * gard_1
        manual_v_1 = BETA2 * manual_v_1 + (1 - BETA2) * gard_1 * gard_1
        manual_m_1_esti = manual_m_1/(1 - BETA1 ** (epoch+1))
        manual_v_1_esti = manual_v_1/(1 - BETA2 ** (epoch+1))
        manual_data_1 = data_1 - 0.01 * manual_m_1_esti / (torch.sqrt(torch.abs(manual_v_1_esti))+1e-08)

        manual_m_2 = BETA1 * manual_m_2 + (1 - BETA1) * gard_2
        manual_v_2 = BETA2 * manual_v_2 + (1 - BETA2) * gard_2 * gard_2
        manual_m_2_esti = manual_m_2/(1 - BETA1 ** (epoch+1))
        manual_v_2_esti = manual_v_2/(1 - BETA2 ** (epoch+1))
        manual_data_2 = data_2 - 0.01 * manual_m_2_esti / (torch.sqrt(torch.abs(manual_v_2_esti))+1e-08)

        print(f' -{epoch}','---' * 19)
        print('权重（手动）w1：',manual_data_1)
        print('权重（手动）w2：',manual_data_2)
        optimizer.step()
        print('权重（更新后）w1：',model.conv_1.weight.data)
        print('权重（更新后）w2：',model.conv_2.weight.data)


        print('==='*20,'\n')
        # break



