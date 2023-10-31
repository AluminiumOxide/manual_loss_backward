import torch
from torch import nn
import torch.optim as optim
import sys
# import graphviz


class SimpleLayer(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(SimpleLayer, self).__init__()
        self.fc_1 = nn.Linear(channel_in, int(channel_in*0.5), bias=False)
        self.fc_2 = nn.Linear(int(channel_in*0.5), channel_out, bias=False)

    def forward(self, x0):
        x1 = self.fc_1(x0)
        x2 = self.fc_2(x1)
        y2 = torch.sigmoid(x2)
        return [x1,x2,y2]


if __name__ == '__main__':
    input = torch.randn(1, 1, 1, 4)
    model = SimpleLayer(4,1)
    target = torch.tensor([1]).view(1,1,1,-1).to(torch.float32)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)


    print(input.size())
    print(model.fc_1.weight.size())
    print(model.fc_2.weight.size())
    print(target.size())

    # sys.exit()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output[-1], target)

        print(f'输入 {input.data} 输出 {output[-1].data} 目标 {target.data} 损失 {loss.item()}')

        w1 = model.fc_1.weight
        w2 = model.fc_2.weight

        x0 = input.data  # >fc>x1  w1
        x1 = output[0].data   # >fc>x2   w2
        x2 = output[1].data   # >sig>y2
        y2 = output[2].data   # >out

        dy2_dx2 = y2 * (1 - y2)
        dx2_dx1 = w2
        dx1_dx0 = w1

        dx2_dw2 = x1
        dx1_dw1 = x0

        # 后面开始拼接
        dz_dw2 = dy2_dx2 * dx2_dw2

        reshape_dy2_dx1 = (dy2_dx2 * dx2_dx1).view(2, 1)
        reshape_dx1_dw1 = dx1_dw1.view(1, 4)
        dz_dw1 = torch.matmul(reshape_dy2_dx1 , reshape_dx1_dw1)

        err = (output[-1].data-target.data)*2

        grad_w2 = err * dz_dw2
        grad_w1 = err * dz_dw1

        loss.backward()
        print('梯度（自动）w1：',model.fc_1.weight.grad)
        print('梯度（自动）w2：',model.fc_2.weight.grad)
        print('梯度 (手动)w1：',grad_w1)
        print('梯度 (手动)w2：',grad_w2)

        print('权重（更新前）：',model.fc_1.weight.data)
        print('权重（更新前）：',model.fc_2.weight.data)
        # new_data = model.fc.weight.data[0, 0]  - 0.01 * model.fc.weight.grad[0, 0]
        new_data_1 = model.fc_1.weight.data  - 0.01 * grad_w1
        new_data_2 = model.fc_2.weight.data  - 0.01 * grad_w2
        print('权重（手动）w1：',new_data_1)
        print('权重（手动）w2：',new_data_2)
        optimizer.step()
        print('权重（更新后）w1：',model.fc_1.weight.data)
        print('权重（更新后）w2：',model.fc_2.weight.data)

        print('---'*20)
        break





