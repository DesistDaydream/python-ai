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
with open(Path(base_model_path) / "chat_template.jinja") as f:
    tokenizer.chat_template = f.read()

# =============================================
# ================ 第一阶段：CPT ===============
# =============================================
cpt_dataset = [
    "DesistDaydream 是个超人，可以上天、下海、入地，甚至可以飞到宇宙边缘。",
    "DesistDaydream 会游泳。",
    "DesistDaydream 会潜水。",
]

cpt_inputs = []
for text in cpt_dataset:
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    inp["labels"] = inp["input_ids"].clone()  # type: ignore
    cpt_inputs.append(inp)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
best_loss = float("inf")
best_model_state = None

print("=== 第一阶段：CPT 训练 ===")
for epoch in range(20):
    epoch_loss = 0.0
    for inp in cpt_inputs:
        res = model.forward(**inp)
        loss: torch.FloatTensor = res["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(cpt_inputs)
    print(f"Epoch {epoch + 1}, 平均 Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# 让模型加载第一阶段训练中，最佳的模型状态，作为第二阶段训练的模型权重。
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"CPT 完成，最低 Loss: {best_loss:.4f}")

# =============================================
# ================ 第二阶段：SFT ===============
# =============================================
sft_dataset = [
    [
        {"role": "user", "content": "DesistDaydream 是谁?"},
        {
            "role": "assistant",
            "content": "DesistDaydream 是个超人，可以上天、入地、下海，甚至可以飞到宇宙边缘。",
        },
    ],
    [
        {"role": "user", "content": "DesistDaydream 打洞吗?"},
        {
            "role": "assistant",
            "content": "会，DesistDaydream 可以入地。",
        },
    ],
    [
        {"role": "user", "content": "DesistDaydream 会游泳吗?"},
        {"role": "assistant", "content": "会，DesistDaydream 可以下海、潜水。"},
    ],
    [
        {"role": "user", "content": "介绍一下 DesistDaydream"},
        {
            "role": "assistant",
            "content": "DesistDaydream 是个超人，可以上天、入地、下海，甚至可以飞到宇宙边缘。",
        },
    ],
]


def generate_labels(input_ids, assistant_mask):
    labels = input_ids.clone()
    labels[assistant_mask == 0] = -100
    return labels


sft_inputs = []
for conversations in sft_dataset:
    inp = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        return_assistant_tokens_mask=True,
    ).to(model.device)  # type: ignore
    inp["labels"] = generate_labels(inp["input_ids"], inp["assistant_masks"])
    sft_inputs.append(inp)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
best_loss = float("inf")
best_model_state = None

print("=== 第二阶段：SFT 训练 ===")
for epoch in range(20):
    epoch_loss = 0.0
    for inp in sft_inputs:
        res = model.forward(**inp)
        loss: torch.FloatTensor = res["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(sft_inputs)
    print(f"Epoch {epoch + 1}, 平均 Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

if best_model_state is not None:
    print(f"--- 训练完成，正在保存 Loss 最低({best_loss:.4f})的模型权重 ---")
    model.load_state_dict(best_model_state)
    save_model(model, output_model_file)
