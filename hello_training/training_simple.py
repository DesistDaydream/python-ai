from torch import optim
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM
from safetensors.torch import save_model

base_model_path = r"D:\appdata\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
output_model_path = r"D:\appdata\models\desistdaydream\model.safetensors"

tokenizer = Qwen2Tokenizer.from_pretrained(base_model_path)
model = Qwen3ForCausalLM.from_pretrained(base_model_path)

# TODO: 这是做什么用的？
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

dataSet = "DesistDaydream 是个超人，可以上天、下海、入地，甚至可以飞到宇宙边缘。"
inputs = tokenizer(dataSet, return_tensors="pt")

# TODO: inputs 里的 labels 是什么，有什么用？
inputs["labels"] = inputs["input_ids"].clone()
print(inputs)

# TODO: 这是一个什么过程？
for i in range(10):
    res = model(**inputs)
    loss = res["loss"]
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # TODO: 添加评估，只有评估结果比上一次好，才保存模型
    save_model(model, output_model_path)
