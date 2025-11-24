import torch
from torch import nn

# TODO: AI 补充出来的注释，需要尝试理解
# 一、定义模型结构
# 1.1 词嵌入矩阵（Embedding Matrix）
# 词嵌入矩阵是一个 lookup table，用于将 Token ids 映射到对应的 token embeddings。
# 每个 Token id 对应矩阵中的一行，这行就是该 Token id 的 token embedding。
# 例如，假设我们有一个包含 1000 个 Token 的词汇表，每个 Token 用一个 50 维的向量表示。
# 那么词嵌入矩阵就是一个 1000x50 的矩阵，每个元素都是一个随机初始化的向量。
# 当模型接收一个 Token id 作为输入时，它会从词嵌入矩阵中查找对应的向量，作为该 Token id 的 token embedding。
# 1.2 前向传播（Forward Pass）
# 前向传播是指模型接收输入数据（如 Token ids），通过层与层之间的计算，最终输出预测结果（如 token embeddings）的过程。
# 在我们的模型中，前向传播的过程如下：
# 1. 接收输入的 Token ids。
# 2. 从词嵌入矩阵中查找对应的 token embeddings。
# 3. 返回这些 token embeddings 作为模型的输出。

# 代码来源: https://www.bilibili.com/video/BV1MAUsBME6f
# 定义一个简单的模型，用于将输入的 Token ids 转换为 token embeddings
class ShankeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # ################ 核心逻辑 ################
        # **初始化**一个模型，本质上是初始化一个矩阵
        # 模型经过多年演进，变成了最简单的模样：随机的矩阵。i.e. Embedding Matrix(嵌入矩阵)
        # 通过 “词表大小” 与 “维度大小” 初始化 词嵌入矩阵
        self.embedding_matrix = nn.Embedding(vocab_size, embedding_dim)

    # 前向传播。必要的实现，不实现将会报错: NotImplementedError: Module [ShankeModel] is missing the required "forward" function
    def forward(self, ids):
        return self.embedding_matrix(ids)

# 定义词表大小和词嵌入维度
vocab_size = 50275
embedding_dim = 768
model = ShankeModel(vocab_size,embedding_dim)

# 随便指定一些 Token ids
ids = torch.tensor([1,3,5])

# 调用模型
token_embeddings = model(ids)
print(token_embeddings)
