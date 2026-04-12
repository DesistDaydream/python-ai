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

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

dataSet = [
    "DesistDaydream 是个超人，可以上天、下海、入地，甚至可以飞到宇宙边缘。",
    "DesistDaydream 会游泳",
    "DesistDaydream 会潜水",
]


all_inputs = []
for text in dataSet:
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids["labels"] = input_ids["input_ids"].clone()  # type: ignore
    all_inputs.append(input_ids)


best_loss = float("inf")

for epoch in range(50):
    epoch_loss = 0.0
    for input_ids in all_inputs:
        res = model.forward(**input_ids)
        loss: torch.FloatTensor = res["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(all_inputs)
    print(f"Epoch {epoch + 1}, 平均 Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

if best_model_state is not None:
    print(f"--- 训练完成，正在保存 Loss 最低({best_loss:.4f})的模型权重 ---")
    model.load_state_dict(best_model_state)
    save_model(model, output_model_file)
