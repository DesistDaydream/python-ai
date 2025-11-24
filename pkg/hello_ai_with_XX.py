import torch
from torch import nn

# 一、定义模型结构
# Linear() 可以暂时理解为使用 Linear 模型，可以假设模型是 y = xA^T + b；
# 10, 1 可以理解为 超参数。i.e. 输入维度为 10，输出维度为 1。输入为几最后生成的模型中的权重就有几个值。
fc = nn.Linear(10, 1)

nn.Embedding

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


model = MyModel()

inputs = torch.rand(16, 10)
outputs = model(inputs)

print(outputs)

print(outputs.size())



# 二、定义损失函数、优化器
# 略

# 三、训练模型
# 注意：fc.state_dict() 并不是真正意义上的训练模型。仅是获取模型的当前参数（e.g. 权重值、etc.）这些参数可能是刚初始化的（随机值），也可能是已经训练过的。
# 通常，在训练完成后调用 state_dict() 来保存模型的参数。这样可以在之后加载这些参数，继续训练或进行推理。
# model = fc.state_dict()

# # 四、保存模型
# # 将训练结果 fc 保存到模型文件 hello_world.pth 中
# torch.save(model, "./models/hello_world.pth")

# # 从 hello_world.pth 模型文件中读取参数
# weight = torch.load("./models/hello_world.pth", weights_only=True)

# 模型文件中的内容本质上是一系列权重值的集合，效果如下：
# OrderedDict({'weight': tensor([[0.0382, -0.1313,  0.2224, -0.2967, -0.2892, -0.2951,  0.0455, -0.0702,
#          -0.2919,  0.2825]]), 'bias': tensor([-0.2147])})
print(list(model.parameters()))
