import torch
from torch import nn
import torch.optim as optim
import sys
import graphviz


def draw_forward(input,output,target,model):
    dot = graphviz.Digraph()
    dot.graph_attr['rankdir'] = 'LR'
    dot.attr(splines='false')  # 禁用曲线布局
    dot.attr(ranksep='3.0')  # 设置子图之间的距离
    dot.attr(nodesep='0.5')  # 设置节点之间的距离
    # 输入层的参数
    with dot.subgraph() as sub_dot:
        sub_dot.attr('node', shape='circle', rank='same')
        for idx_in in range(input.size()[-1]):
            sub_dot.node(f'A{idx_in:d}', label=f'{input[0, 0, 0, idx_in]:.2f}')
    # 输出层的参数
    with dot.subgraph() as sub_dot:
        sub_dot.attr('node', shape='circle', rank='same')
        for idx_out in range(output.size()[-1]):
            sub_dot.node(f'B{idx_out:d}', label=f'{output[0, 0, 0, idx_out]:.2f}')
    # 目标层的参数
    with dot.subgraph() as sub_dot:
        sub_dot.attr('node', shape='circle', rank='same')
        for idx_out in range(target.size()[-1]):
            sub_dot.node(f'C{idx_out:d}', label=f'{target[0, 0, 0, idx_out]:.2f}')
    # layer的权重作为连接线
    for idx_li_1 in range(model.fc.weight.size()[0]):  # 1
        for idx_li_2 in range(model.fc.weight.size()[1]):  # 4
            # dot.edge(f'A{idx_li_2:d}', f'B{idx_li_1:d}',splines='false')
            dot.edge(f'A{idx_li_2:d}', f'B{idx_li_1:d}',
                     label=f'{model.fc.weight[idx_li_1, idx_li_2]:.2f}')  # {idx_li_1},{idx_li_2} :
    for idx_loss in range(target.size()[-1]):
        # dot.edge(f'A{idx_li_2:d}', f'B{idx_li_1:d}',splines='false')
        dot.edge(f'B{idx_loss:d}', f'C{idx_loss:d}', )
        # label=f'{idx_li_1},{idx_li_2} : {model.fc.weight[idx_li_1,idx_li_2]:.2f}')
    return dot


class SimpleLayer(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(SimpleLayer, self).__init__()
        self.fc = nn.Linear(channel_in, channel_out, bias=False)

    def forward(self, x):
        x1 = self.fc(x)
        return x1


if __name__ == '__main__':
    input = torch.randn(1, 1, 1, 4)
    model = SimpleLayer(4,1)
    output = model(input)
    target = torch.tensor([1]).view(1,1,1,-1).to(torch.float32)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)


    print(input.size())
    print(model.fc.weight.size())
    print(output.size())
    print(target.size())

    # sys.exit()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        print(f'输入 {input.data} 输出 {output.data} 目标 {target.data} 损失 {loss.item()}')
        # print(model.fc.weight.grad)

        dx_dw = input.data
        loss.backward()

        dot = draw_forward(input,output,target,model)
        dot.render('img_1_linear', format='png')

        print('梯度（自动）：',model.fc.weight.grad)
        print('梯度 (手动)：', 2 * (output.data-target.data) * dx_dw )  # input.data 相当于d_output/d_w ,这里变成 2(out-tar)是因为 MSE2求导

        print('权重（更新前）：',model.fc.weight)
        new_data = model.fc.weight.data  - 0.01 * model.fc.weight.grad
        print('权重（手动）：',new_data)
        optimizer.step()
        print('权重（更新后）：',model.fc.weight)



        break

