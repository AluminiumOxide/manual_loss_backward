import torch
from torch import nn
import torch.optim as optim
import sys
import graphviz
from pyvis.network import Network

def draw_forward(x0, x1, y1, x2, y2, w1, w2, target):
    dot = graphviz.Digraph('G', node_attr={'shape': 'plaintext'})
    dot.graph_attr['rankdir'] = 'LR'

    label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR><TD COLSPAN="4">X_0</TD></TR>
    <TR><TD width="50" height="50" PORT="x0_00">{x0[0,0,0,0]:.2f}</TD>
        <TD width="50" height="50" PORT="x0_01">{x0[0,0,0,1]:.2f}</TD>
        <TD width="50" height="50" PORT="x0_02">{x0[0,0,0,2]:.2f}</TD>
        <TD width="50" height="50" PORT="x0_03">{x0[0,0,0,3]:.2f}</TD></TR>
    <TR><TD width="50" height="50" PORT="x0_10">{x0[0,0,1,0]:.2f}</TD>
        <TD width="50" height="50" PORT="x0_11">{x0[0,0,1,1]:.2f}</TD>
        <TD width="50" height="50" PORT="x0_12">{x0[0,0,1,2]:.2f}</TD>
        <TD width="50" height="50" PORT="x0_13">{x0[0,0,1,3]:.2f}</TD></TR>
    <TR><TD width="50" height="50" PORT="x0_20">{x0[0,0,2,0]:.2f}</TD>
        <TD width="50" height="50" PORT="x0_21">{x0[0,0,2,1]:.2f}</TD>
        <TD width="50" height="50" PORT="x0_22">{x0[0,0,2,2]:.2f}</TD>
        <TD width="50" height="50" PORT="x0_23">{x0[0,0,2,3]:.2f}</TD></TR>
    <TR><TD width="50" height="50" PORT="x0_30">{x0[0,0,3,0]:.2f}</TD>
        <TD width="50" height="50" PORT="x0_31">{x0[0,0,3,1]:.2f}</TD>
        <TD width="50" height="50" PORT="x0_32">{x0[0,0,3,2]:.2f}</TD>
        <TD width="50" height="50" PORT="x0_33">{x0[0,0,3,3]:.2f}</TD></TR>
    </TABLE>>'''
    dot.node('x0', label, shape='square', width='1', height='1')

    label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR><TD COLSPAN="4">W_1</TD></TR>
    <TR><TD width="50" height="50" PORT="w1_00">{w1[0,0,0,0]:.2f}</TD>
        <TD width="50" height="50" PORT="w1_01">{w1[0,0,0,1]:.2f}</TD>
        <TD width="50" height="50" PORT="w1_02">{w1[0,0,0,2]:.2f}</TD></TR>
    <TR><TD width="50" height="50" PORT="w1_10">{w1[0,0,1,0]:.2f}</TD>
        <TD width="50" height="50" PORT="w1_11">{w1[0,0,1,1]:.2f}</TD>
        <TD width="50" height="50" PORT="w1_12">{w1[0,0,1,2]:.2f}</TD></TR>
    <TR><TD width="50" height="50" PORT="w1_20">{w1[0,0,2,0]:.2f}</TD>
        <TD width="50" height="50" PORT="w1_21">{w1[0,0,2,1]:.2f}</TD>
        <TD width="50" height="50" PORT="w1_22">{w1[0,0,2,2]:.2f}</TD></TR>
    </TABLE>>'''
    dot.node('w1', label, group='same')

    label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR><TD COLSPAN="4">X_1</TD></TR>
    <TR><TD width="50" height="50" PORT="x1_00">{x1[0,0,0,0]:.2f}</TD>
        <TD width="50" height="50" PORT="x1_01">{x1[0,0,0,1]:.2f}</TD></TR>
    <TR><TD width="50" height="50" PORT="x1_10">{x1[0,0,1,0]:.2f}</TD>
        <TD width="50" height="50" PORT="x1_11">{x1[0,0,1,1]:.2f}</TD></TR>
    </TABLE>>'''
    dot.node('x1', label, shape='square', width='1', height='1')

    label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR><TD COLSPAN="4">Y-1</TD></TR>
    <TR><TD width="50" height="50" PORT="y1_00">{y1[0,0,0,0]:.2f}</TD>
        <TD width="50" height="50" PORT="y1_01">{y1[0,0,0,1]:.2f}</TD></TR>
    <TR><TD width="50" height="50" PORT="y1_10">{y1[0,0,1,0]:.2f}</TD>
        <TD width="50" height="50" PORT="y1_11">{y1[0,0,1,1]:.2f}</TD></TR>
    </TABLE>>'''
    dot.node('y1', label, shape='square', width='1', height='1')

    label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR><TD COLSPAN="4">W_2</TD></TR>
    <TR><TD width="50" height="50" PORT="w2_00">{w2[0,0,0,0]:.2f}</TD>
        <TD width="50" height="50" PORT="w2_01">{w2[0,0,0,1]:.2f}</TD></TR>
    <TR><TD width="50" height="50" PORT="w2_10">{w2[0,0,1,0]:.2f}</TD>
        <TD width="50" height="50" PORT="w2_11">{w2[0,0,1,1]:.2f}</TD></TR>
    </TABLE>>'''
    dot.node('w2', label)

    label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR><TD COLSPAN="4">X_2</TD></TR>
    <TR><TD width="50" height="50" PORT="x2_00">{x2[0,0,0,0]:.2f}</TD></TR>
    </TABLE>>'''
    dot.node('x2', label, shape='square', width='1', height='1')

    label = f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR><TD COLSPAN="4">Y_2</TD></TR>
    <TR><TD width="50" height="50" PORT="y2_00">{y2[0,0,0,0]:.2f}</TD></TR>
    </TABLE>>'''
    dot.node('y2', label, shape='square', width='1', height='1')
    dot.node('target', label=f'{target[0, 0, 0, 0]:.2f}', shape='circle',)

    dot.edge('x0','x1',label='dx1/dx0')
    dot.edge('w1','x1',label='dx1/dw1')
    dot.edge('x1','y1',label='dy1/dx1')
    dot.edge('y1','x2',label='dx2/dy1')
    dot.edge('w2','x2',label='dx2/dw2')
    dot.edge('x2','y2',label='dy2/dx2')
    dot.edge('y2','target', style='dotted')

    return dot


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


def manual_conv(input, weight):
    batch_size, channels, height, width = input.size()
    kernel_size = weight.size(2)  # 获取卷积核的形状信息
    padding = 0
    stride = 1
    # 计算卷积后的输出形状
    output_height = (height - kernel_size + 2 * padding) // stride + 1
    output_width = (width - kernel_size + 2 * padding) // stride + 1
    # 创建输出张量
    manual_x1 = torch.zeros(batch_size, 1, output_height, output_width)

    d_x1 = torch.zeros([output_height, output_width,
                        input.size(2),input.size(3)])
    d_w1 = torch.zeros([output_height, output_width,
                        weight.size(2),weight.size(3)])
    # 执行卷积操作
    for i in range(output_height):
        for j in range(output_width):
            manual_x1[0, 0, i, j] = torch.sum(input[0, 0, i:i + kernel_size, j:j + kernel_size] * weight)
    """
    对于单层的来说,
    每个w，每个位置了挪了多少个，相当于 output_height × output_width 次乘法计算，现在要做的就是把这些位置的内容求和，你就能得到这里w的梯度了
    对于x，每个位置经历多少次卷积核的计算，把W求和，就能得到这里x的梯度了
    """
    for hy in range(manual_x1.size(2)):  # h_out 对应输出的高度
        for wy in range(manual_x1.size(3)):  # w_out 对应输出的宽度
            for hw in range(kernel_size):  # H 对应W的高度
                for ww in range(kernel_size):  # W 对应W的宽度
                    d_w1[hy, wy, hw, ww] = input[0, 0, hy+hw, wy+ww]


    for i in range(output_height):  # H
        for j in range(output_width):  # W
            d_x1[i, j, i:i + kernel_size, j:j + kernel_size] += weight[0,0,0:kernel_size, 0:kernel_size]

    return [manual_x1,d_x1,d_w1]


if __name__ == '__main__':
    input = torch.reshape(torch.arange(1,17), (1, 1, 4, 4))/10
    model = SimpleLayer(1,1,1)
    target = torch.tensor([1]).view(1,1,1,-1).to(torch.float32)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output[-1], target)

        print(f'输入 {input.data} 输出 {output[-1].data} 目标 {target.data} 损失 {loss.item()}')

        w1 = model.conv_1.weight.data  # 1,1,3,3
        w2 = model.conv_2.weight.data  # 1,1,3,3
        x0 = input.data
        x1 = output[0].data  # x0 > conv > x1 with w1
        y1 = output[1].data  # x1 > sig > y1
        x2 = output[2].data   # y1 > conv > x2 with w2
        y2 = output[3].data   # x2 > sig > y2

        dot = draw_forward(x0, x1, y1, x2, y2, w1, w2, target)
        dot.render('img_5_conv', format='png')

        # 输出 y=f(x,w) dy/dx dy/dw
        dy2_x2 = y2*(1-y2)
        [manual_x2,dx2_y1,dx2_w2] = manual_conv(y1,w2)
        dy1_x1 = y1*(1-y1)
        [manual_x1,dx1_x0,dx1_w1] = manual_conv(x0,w1)

        dy2_w2 = dy2_x2 * dx2_w2

        dy2_x1 = dy2_x2 * dx2_y1 * dy1_x1
        dy2_x1_reshape = dy2_x1.view(4)
        dx1_w1_reshape = dx1_w1.view(4, 9)
        dy2_w1 = torch.matmul(dy2_x1_reshape, dx1_w1_reshape).view(3, 3)

        d_err = (output[-1].data - target.data) * 2

        grad_w2 = d_err * dy2_w2
        grad_w1 = d_err * dy2_w1

        loss.backward()
        print('梯度（自动）w2：',model.conv_2.weight.grad)
        print('梯度 (手动)w2：',grad_w2)
        print('梯度（自动）w1：',model.conv_1.weight.grad)
        print('梯度 (手动)w1：',grad_w1)

        print('权重（更新前）：',model.conv_1.weight.data)
        print('权重（更新前）：',model.conv_2.weight.data)
        new_data_1 = model.conv_1.weight.data  - 0.01 * grad_w1
        new_data_2 = model.conv_2.weight.data  - 0.01 * grad_w2
        print('权重（手动）w1：',new_data_1)
        print('权重（手动）w2：',new_data_2)
        optimizer.step()
        print('权重（更新后）w1：',model.conv_1.weight.data)
        print('权重（更新后）w2：',model.conv_2.weight.data)

        print('---'*20)
        break





