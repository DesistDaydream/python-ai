from pathlib import Path
import torch
import transformers
from safetensors.torch import save_model


# base_model_path = r"D:\appdata\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
base_model_path = "/mnt/d/appdata/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/"
# output_model_file = r"D:\appdata\models\desistdaydream\model.safetensors"
output_model_file = "/mnt/d/appdata/models/desistdaydream/model.safetensors"

# 实例化 分词器 与 模型
tokenizer: transformers.Qwen2Tokenizer = transformers.Qwen2Tokenizer.from_pretrained(
    Path(base_model_path)
)
model: transformers.Qwen3ForCausalLM = transformers.Qwen3ForCausalLM.from_pretrained(
    Path(base_model_path),
    torch_dtype="auto",
    device_map="auto",
)
# 修复千问系列模型的 chat template BUG:  but chat template does not contain {% generation %} keyword.
with open(Path(base_model_path) / "chat_template.jinja") as f:
    tokenizer.chat_template = f.read()

# =================================================
# ================ 一、数据集预处理 ================
# =================================================
# ================ ！！！注意！！！================
# 不可以只准备一条数据，否则必然会出现“过拟合”，只有一个样本，模型会“记忆”这个样本，而不是学习一般规律。
# i.e. 训练完成后，只有输入 "DesistDaydream是谁?"，模型才会返回关联的描述。否则，模型会返回随机内容。
# ================================================
prompt = "DesistDaydream 是个超人，可以上天、入地、下海，甚至可以飞到宇宙边缘。"
conversations_list = [
    [
        {"role": "user", "content": "DesistDaydream 是谁?"},
        {"role": "assistant", "content": prompt},
    ],
    [
        {"role": "user", "content": "你知道 DesistDaydream 吗?"},
        {"role": "assistant", "content": "知道，" + prompt},
    ],
    [
        {"role": "user", "content": "介绍一下 DesistDaydream"},
        {"role": "assistant", "content": prompt},
    ],
]


# TODO:只对 assistant 的回答部分计算 loss，user 的问题部分被屏蔽掉了（设为 -100）。
def generate_labels(input_ids, assistant_mask):
    labels = input_ids.clone()
    labels[assistant_mask == 0] = -100
    return labels


# 将文本数据集转为模型可以接受的 Tensor。
all_input_ids = []
for conversations in conversations_list:
    input_ids = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        return_assistant_tokens_mask=True,
    ).to(model.device)  # type: ignore
    input_ids["labels"] = generate_labels(
        input_ids["input_ids"], input_ids["assistant_masks"]
    )
    all_input_ids.append(input_ids)
# 经过 generate_labels 处理后，input_ids 通常包含：
# - input_ids # TokenIDs 序列
# - attention_mask # 注意力掩码，用于屏蔽 padding token
# - labels # 监督信号，告诉模型“正确答案是什么”。在 SFT 场景下，用于屏蔽用户的输入部分，i.e. 设为 -100。


# ==================================================
# 实例化 优化器。
# 优化器用于更新模型参数，使模型输出更接近目标。
# 优化器可以计算模型参数的梯度，然后根据梯度更新参数。
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# 初始化一个最高损失，用于后续对比保存
best_loss = float("inf")
best_model_state = None
# ==================================================
# ================ 二、开始 SFT 训练 ================
# ==================================================
for epoch in range(5):
    epoch_loss = 0.0
    for input_ids in all_input_ids:
        # ================ 计算损失 ================
        # 一般都这么用。
        # res = model(**inputs)
        # 我是为了看具体属性，指定了 forward() 方法
        # 使用 model.__call__() 即可发现，__call__ 方法并不在 Qwen3ForCausalLM 中，
        # 而是按照继承链往上走，最终来自 PyTorch 的 nn.Module 的 __call__ 方法
        # 一路追 nn.Module.__call__ 的逻辑，本质是调用了 self.forward() 方法
        #
        # 前向传播。模型做一次完整的 "预测 + 算分"；
        res = model.forward(**input_ids)
        # 获取计算结果中的损失，损失表示模型预测得有多错，这个值越小说明模型越接近训练目标。值过小将会导致模型过拟合。
        loss: torch.FloatTensor = res["loss"]
        # ================ 重置梯度 ================
        # 避免累加
        optimizer.zero_grad()
        # ================ 反向传播 ================
        # 计算出每个参数对 loss 的影响方向和大小（即梯度）
        loss.backward()
        # ================ 更新参数 ================
        # 根据梯度，更新模型权重
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(all_input_ids)
    print(f"Epoch {epoch + 1}, 平均 Loss: {avg_loss:.4f}")

    # 如果当前 Loss 是历史最低，记录下此时的模型状态
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        # best_model_state = copy.deepcopy(model.state_dict()) # copy.deepcopy 会在内存中存一份当前的参数快照
        # TODO: {k: v.cpu().clone() for k, v in model.state_dict().items()} 和 copy.deepcopy(model.state_dict()) 有什么区别？

# TODO: 如何判断模型有没有过拟合？

# =============================================
# ================ 三、保存模型 ================
# =============================================
# 训练全部完成后，加载效果最好的那次参数，并执行保存
if best_model_state is not None:
    print(f"--- 训练完成，正在保存 Loss 最低({best_loss:.4f})的模型权重 ---")
    model.load_state_dict(best_model_state)
    save_model(model, output_model_file)
