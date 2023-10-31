import torch
from torch import nn
import torch.optim as optim
import sys
import graphviz


class SimpleLayer(nn.Module):
    def __init__(self):
        super(SimpleLayer, self).__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


if __name__ == '__main__':
    input = torch.randn(1, 1, 1, 8)
    model = SimpleLayer()
    output = model(input)
    target = torch.arange(2).view(1,1,1,2).to(torch.float32)
    # criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01)


    print(input.size())
    print(model.fc.weight.size())
    print(output.size())
    print(target.size())


    dot = graphviz.Digraph()
    dot.graph_attr['rankdir'] = 'LR'

    dot.attr(splines='false')# 禁用曲线布局
    dot.attr(ranksep='3.0')  # 设置子图之间的距离
    dot.attr(nodesep='0.5')  # 设置节点之间的距离
    # 输入层的参数
    with dot.subgraph() as sub_dot:
        sub_dot.attr('node',shape='circle',rank='same')
        for idx_in in range(input.size()[-1]):
            sub_dot.node(f'A{idx_in:d}', label=f'{input[0,0,0,idx_in]:.2f}')

    # 输出层的参数
    with dot.subgraph() as sub_dot:
        sub_dot.attr('node',shape='circle',rank='same')
        for idx_out in range(output.size()[-1]):
            sub_dot.node(f'B{idx_out:d}', label=f'{output[0,0,0,idx_out]:.2f}')


    # 目标层的参数
    with dot.subgraph() as sub_dot:
        sub_dot.attr('node',shape='circle',rank='same')
        for idx_out in range(target.size()[-1]):
            sub_dot.node(f'C{idx_out:d}', label=f'{target[0,0,0,idx_out]:.2f}')


    # layer的权重作为连接线
    for idx_li_1 in range(model.fc.weight.size()[0]):  # 1
        for idx_li_2 in range(model.fc.weight.size()[1]):  # 8
            # dot.edge(f'A{idx_li_2:d}', f'B{idx_li_1:d}',splines='false')
            dot.edge(f'A{idx_li_2:d}', f'B{idx_li_1:d}',
                     label=f'{idx_li_1},{idx_li_2} : {model.fc.weight[idx_li_1,idx_li_2]:.2f}')

    #
    for idx_loss in range(target.size()[-1]):
        # dot.edge(f'A{idx_li_2:d}', f'B{idx_li_1:d}',splines='false')
        dot.edge( f'B{idx_loss:d}',f'C{idx_loss:d}',)
                 # label=f'{idx_li_1},{idx_li_2} : {model.fc.weight[idx_li_1,idx_li_2]:.2f}')


    dot.render('demo_3', format='png')








