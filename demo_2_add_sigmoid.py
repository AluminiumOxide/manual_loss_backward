import torch
from torch import nn
import torch.optim as optim
import sys
import graphviz

def draw_forward(input,x1,y1,target,model):
    dot = graphviz.Digraph('G')
    dot.graph_attr['rankdir'] = 'LR'
    dot.attr(splines='false')  # 禁用曲线布局
    dot.attr(ranksep='3.0')  # 设置子图之间的距离
    dot.attr(nodesep='0.5')  # 设置节点之间的距离
    # 输入层的参数
    with dot.subgraph(name='input') as sub_dot:
        sub_dot.attr('node', shape='circle', rank='same')
        for idx_in in range(input.size()[-1]):
            sub_dot.node(f'A{idx_in:d}', label=f'{input[0, 0, 0, idx_in]:.2f}')
    # 中间层的参数
    with dot.subgraph(name='hid') as sub_dot:
        sub_dot.attr('node', label='x1', shape='circle', rank='same')
        for idx_out in range(x1.size()[-1]):
            sub_dot.node(f'B{idx_out:d}', label=f'{x1[0, 0, 0, idx_out]:.2f}')
    # 输出层的参数
    with dot.subgraph() as sub_dot:
        sub_dot.attr('node',label='y1', shape='circle', rank='same')
        for idx_out in range(y1.size()[-1]):
            sub_dot.node(f'C{idx_out:d}', label=f'{y1[0, 0, 0, idx_out]:.2f}')
    # 目标层的参数
    with dot.subgraph() as sub_dot:
        sub_dot.attr('node', shape='circle', rank='same')
        for idx_out in range(target.size()[-1]):
            sub_dot.node(f'D{idx_out:d}', label=f'{target[0, 0, 0, idx_out]:.2f}')
    # layer的权重作为连接线
    for idx_li_1 in range(model.fc.weight.size()[0]):  # 1
        for idx_li_2 in range(model.fc.weight.size()[1]):  # 4
            # dot.edge(f'A{idx_li_2:d}', f'B{idx_li_1:d}',splines='false')
            dot.edge(f'A{idx_li_2:d}', f'B{idx_li_1:d}',
                     label=f'{model.fc.weight[idx_li_1, idx_li_2]:.2f}')  # {idx_li_1},{idx_li_2} :
    for idx_loss in range(target.size()[-1]):
        dot.edge(f'B{idx_loss:d}', f'C{idx_loss:d}', label = 'x1 > sigmoid > y1')
    for idx_loss in range(target.size()[-1]):
        dot.edge(f'C{idx_loss:d}', f'D{idx_loss:d}',label = '> loss <', style='dotted' )

    return dot


class SimpleLayer(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(SimpleLayer, self).__init__()
        self.fc = nn.Linear(channel_in, channel_out, bias=False)

    def forward(self, x0):
        x1 = self.fc(x0)
        y1 = torch.sigmoid(x1)
        return [x1,y1]


if __name__ == '__main__':
    input = torch.randn(1, 1, 1, 4)
    model = SimpleLayer(4,1)
    output = model(input)
    target = torch.tensor([1]).view(1,1,1,-1).to(torch.float32)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)


    print(input.size())
    print(model.fc.weight.size())
    print(output[-1].size())
    print(target.size())

    # sys.exit()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output[-1], target)


        print(f'输入 {input.data} 输出 {output[-1].data} 目标 {target.data} 损失 {loss.item()}')
        x1 = output[0].data  # y x1
        y1 = output[1].data  # z Y1

        dot = draw_forward(input,x1,y1,target,model)
        dot.render('img_2_add_sigmoid', format='png')


        dy1_dx1 = y1 * (1 - y1)
        dx1_dw1 = input.data

        dy1_dw1 = dy1_dx1 * dx1_dw1

        d_err = (output[-1].data - target.data) * 2

        grad = d_err * dy1_dw1

        loss.backward()
        print('梯度（自动）：',model.fc.weight.grad)
        print('梯度 (手动)：',grad) # 这里相当于 df/dw[i] 这个偏导只是 x[i] 的值，这个值

        print('权重（更新前）：',model.fc.weight.data)
        new_data = model.fc.weight.data  - 0.01 * grad
        print('权重（手动）：',new_data)
        optimizer.step()
        print('权重（更新后）：',model.fc.weight.data)


        break







