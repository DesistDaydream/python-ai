from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen3-0.6B"
model_name = r"D:\appdata\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"

# 加载分词器。从模型目录中查找 merges.txt, vocab.json, tokenizer.json, tokenizer_config.json 文件。
# 分词器可以实现如下两个功能
# - 将用户输入转换为 token 序列。(可读文本 ——> token 序列)
# - 将模型输出转换为可读文本。（token 序列 ——> 可读文本）
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(tokenizer)  # 实例化的分词器来源于 tokenizer.json 和 tokenizer_config.json 文件
# 加载模型。从模型目录中查找 model.safetensors 或 pytorch_model.bin 权重文件，及其相关配置文件
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")


# -------- 准备要输入给模型的文本 --------
# 准备要输入给模型的文本
prompt = "你好"
messages = [
    {"role": "user", "content": prompt},
]
# 使用 Chat Template 渲染消息里列表，以便模型区分。
# 渲染后的文本，包含了特殊的 token，用于表示 用户输入、模型回答、etc.
# 渲染后的效果类似如下：
# <|im_start|>user
# 你好<|im_end|>
text = tokenizer.apply_chat_template(
    messages,  # 要使用模板处理的消息列表
    tokenize=False,  # 是否将文本转换为 TokenID 序列。默认为 False。
    add_generation_prompt=False,  # 是否在文本末尾添加生成提示。默认为 False。
    enable_thinking=False,  # 在思考和非思考模式之间切换。默认为 True。
)
print(text)
# 将渲染后的文本转为包含 Tensor(张量) 的字典。就像这样：
# {'input_ids': tensor([[151644,    872,    198, 108386, 151645,    198]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
print(model_inputs)

# -------- ！！！模型进行推理！！！ --------
# 输入 Tensor，输出 Tensor。这是模型进行计算（i.e. 推理）的过程，也是整个代码，耗费时间最长的一步。
generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
print(generated_ids)
# 去掉用户输入的部分留下模型生成的部分。生成了 TokenID 序列。
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
print(output_ids)

# 直接解码全部输出，不区分 thinking
output_text = tokenizer.decode(
    output_ids,  # 可以换成 generated_ids 看看解码结果
    skip_special_tokens=False,
)
print(output_text)

# 解码时区分 thinking 和 content
# try:
#     # 在 tokenizer.json 文件中可知，151668 对应 </think>
#     index = len(output_ids) - output_ids[::-1].index(151668)
# except ValueError:
#     index = 0
# # 使用分词器解码 thinking 部分的内容
# thinking_content = tokenizer.decode(
#     output_ids[:index],
#     skip_special_tokens=False,
# ).strip("\n")
# print("thinking content:", thinking_content)
# # 使用分词器解码 content 部分的内容
# content = tokenizer.decode(
#     output_ids[index:],
#     skip_special_tokens=False,
# ).strip("\n")
# print("content:", content)
