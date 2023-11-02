import torch
from torch import nn
import torch.optim as optim
import sys
import graphviz
from pyvis.network import Network


def draw_forward(input,x1,y1,x2,y2,target,model):
    dot = graphviz.Digraph('G')
    dot.graph_attr['rankdir'] = 'LR'
    dot.attr(splines='polyline')  # 曲线布局
    dot.attr(ranksep='1')  # 设置子图之间的距离
    dot.attr(nodesep='0.5')  # 设置节点之间的距离
    # 输入层的参数
    with dot.subgraph(name='input') as sub_dot:
        sub_dot.attr('node',  shape='circle', group='same')
        for idx_in in range(input.size()[-1]):
            sub_dot.node(f'A{idx_in:d}', label=f'{input[0, 0, 0, idx_in]:.2f}')
    # x1
    with dot.subgraph() as sub_dot:
        sub_dot.attr('node',  shape='circle', group='same')
        for idx_out in range(x1.size()[-1]):
            sub_dot.node(f'B{idx_out:d}', label=f'{x1[0, 0, 0, idx_out]:.2f}')

    # y1
    with dot.subgraph() as sub_dot:
        sub_dot.attr('node', shape='circle', group='same')
        for idx_out in range(y1.size()[-1]):
            sub_dot.node(f'C{idx_out:d}', label=f'{y1[0, 0, 0, idx_out]:.2f}')

    # x2
    with dot.subgraph() as sub_dot:
        sub_dot.attr('node', shape='circle', group='same')
        for idx_out in range(x2.size()[-1]):
            sub_dot.node(f'D{idx_out:d}', label=f'{x2[0, 0, 0, idx_out]:.2f}')

    # y2
    with dot.subgraph() as sub_dot:
        sub_dot.attr('node', shape='circle', group='same')
        for idx_out in range(y2.size()[-1]):
            sub_dot.node(f'E{idx_out:d}', label=f'{y2[0, 0, 0, idx_out]:.2f}')

    # 目标层的参数
    with dot.subgraph() as sub_dot:
        sub_dot.attr('node', shape='circle', group='same')
        for idx_out in range(target.size()[-1]):
            sub_dot.node(f'F{idx_out:d}', label=f'{target[0, 0, 0, idx_out]:.2f}')

    # layer的权重作为连接线
    # x0连x1
    for idx_li_1 in range(model.fc_1.weight.size()[0]):  # 1
        for idx_li_2 in range(model.fc_1.weight.size()[1]):  # 4
            dot.edge(f'A{idx_li_2:d}', f'B{idx_li_1:d}',label=f'{model.fc_1.weight[idx_li_1, idx_li_2]:.2f}')

    # x1连y1
    for idx_out in range(y1.size()[-1]):
        dot.edge(f'B{idx_out:d}', f'C{idx_out:d}', label='x1 > sigmoid > y1')

    # y1连x2
    for idx_li_1 in range(model.fc_2.weight.size()[0]):  # 1
        for idx_li_2 in range(model.fc_2.weight.size()[1]):  # 4
            dot.edge(f'C{idx_li_2:d}', f'D{idx_li_1:d}',label=f'{model.fc_2.weight[idx_li_1, idx_li_2]:.2f}')

    # x2连y2
    for idx_out in range(y2.size()[-1]):
        dot.edge(f'D{idx_out:d}', f'E{idx_out:d}', label='x2 > sigmoid > y2')

    # y2连tar
    for idx_out in range(target.size()[-1]):
        dot.edge(f'E{idx_out:d}', f'F{idx_out:d}', label='> loss <', style='dotted')

    return dot


class SimpleLayer(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(SimpleLayer, self).__init__()
        self.fc_1 = nn.Linear(channel_in, int(channel_in*0.5), bias=False)
        self.fc_2 = nn.Linear(int(channel_in*0.5), channel_out, bias=False)

    def forward(self, x0):
        x1 = self.fc_1(x0)
        y1 = torch.sigmoid(x1)
        x2 = self.fc_2(y1)
        y2 = torch.sigmoid(x2)
        return [x1,y1,x2,y2]


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
        x0 = input.data
        x1 = output[0].data  # x0 > fc > x1 with w1
        y1 = output[1].data  # x1 > sig > y1
        x2 = output[2].data   # y1 > fc > x2 with w2
        y2 = output[3].data   # x2 > sig > y2

        dot = draw_forward(input,x1,y1,x2,y2,target,model)
        dot.render('img_4_2layer_sig', format='png')

        dy2_x2 = y2*(1-y2)
        dx2_y1 = w2
        dy1_x1 = y1*(1-y1)
        dx1_x0 = w1
        dx1_w1 = x0
        dx2_w2 = y1

        dy2_w2 = dy2_x2 * dx2_w2

        dy2_x1 =  dy2_x2 * dx2_y1 * dy1_x1
        reshape_y2_x1 = (dy2_x1).view(2, 1)
        reshape_x1_w1 = dx1_w1.view(1, 4)
        dy2_w1 = torch.matmul(reshape_y2_x1, reshape_x1_w1)

        d_err = (output[-1].data - target.data) * 2

        grad_w2 = d_err * dy2_w2
        grad_w1 = d_err * dy2_w1

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







# import torch
#
# # 创建两个张量
# tensor1 = torch.tensor([[[[1, 2]]]])  # 形状为(1, 1, 1, 2)
# tensor2 = torch.tensor([[[[1, 2, 3, 4]]]])  # 形状为(1, 1, 1, 4)
#
# # 调整张量的维度
# tensor1 = tensor1.view(2, 1)
# tensor2 = tensor2.view(1, 4)
#
# # 进行矩阵乘法
# result = torch.matmul(tensor1, tensor2)
#
# print(result)