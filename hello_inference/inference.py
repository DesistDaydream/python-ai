# from transformers import Qwen2Tokenizer
from transformers import PreTrainedTokenizerBase
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch  # 导入 torch 库，用于手动创建 Tensor(张量)，而不是使用 transformers 库。

# https://huggingface.co/Qwen/Qwen3-0.6B
# model_name = "Qwen/Qwen3-0.6B"
model_path = r"D:\appdata\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
# model_path = r"D:\appdata\models\desistdaydream"

# ======================================================
# ================ 零、实例化分词器与模型 ================
# ======================================================
# 加载分词器。从模型目录中查找 merges.txt, vocab.json, tokenizer.json, tokenizer_config.json 文件。
# 分词器可以实现如下两个功能
# - 将用户输入转换为 token 序列。(可读文本 ——> token 序列)
# - 将模型输出转换为可读文本。（token 序列 ——> 可读文本）
# ！！！注意：若指定模型不存在，则会下载分词器文件到本地目录。通常包含 config.json, merges.txt, tokenizer.json, tokenizer_config.json, vocab.json
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path)
print(
    type(tokenizer), tokenizer.__class__.__name__
)  # 检查一下类名，可以替换 PreTrainedTokenizerBase 为 Qwen2Tokenizer
# 实例化的分词器来源于 tokenizer.json 和 tokenizer_config.json 文件
# print(tokenizer)
# 加载模型。从模型目录中查找 model.safetensors 或 pytorch_model.bin 权重文件，及其相关配置文件
# 模型加载完成后，根据 device_map 参数，会自动将模型移动到指定的设备（如 GPU 或 CPU）（device_map 参数依赖 accelerate 包）
# 不加 device_map 参数，默认会将模型加载到 CPU 上。
# ！！！注意：若指定模型不存在，则会下载模型文件到本地目录。通常包含 model.safetensors
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)


# 准备要输入给模型的文本
prompt = "hi, i'm DesistDaydream"
messages = [
    {"role": "user", "content": prompt},
]

# ====================================================
# ================ 一、准备推理前的输入 ================
# ====================================================
#
# =========================================================================================================
# 下面这一段，是一套简化的代码，apply_chat_template() 方法只需要传入特定参数，即可完整实现 “一、准备推理前的输入”
# =========================================================================================================
# inputs_ids_with_mask = tokenizer.apply_chat_template(
#     messages,  # 要使用模板处理的消息列表
#     tokenize=True,  # 是否将文本转换为 Token 序列。默认为 Ture。关闭后只输出渲染后的文本。
#     return_tensors="pt",  # 返回 Tensor。只有当 tokenize 为 True 时才有效。
#     add_generation_prompt=True,  # 是否在文本末尾添加生成提示。默认为 False 。
#     enable_thinking=False,  # 在思考和非思考模式之间切换。默认为 True 。
# ).to(model.device)  # type: ignore
# =========================================================================================================
#
# 使用 Chat Template 渲染消息里列表，以便模型区分。
# 渲染后的文本，包含了特殊的 token，用于表示 用户输入、模型回答、etc.
# 渲染后的效果类似如下：
# <|im_start|>user
# hi, i'm DesistDaydream<|im_end|>
# <|im_start|>assistant
# <think>
#
# </think>
text = tokenizer.apply_chat_template(
    messages,  # 要使用模板处理的消息列表
    tokenize=False,  # 是否将文本转换为 Token 序列。默认为 Ture。关闭后只输出渲染后的文本。
    add_generation_prompt=True,  # 是否在文本末尾添加生成提示。默认为 False 。
    enable_thinking=False,  # 在思考和非思考模式之间切换。默认为 True 。
)
print(text)

# ================================================
# 根据词表（merges.txt）分词，生成 Token sequence。
token_sequence = tokenizer.tokenize(text)  # type: ignore
print(token_sequence)
# 根据词表（vocab.json）将 Token sequence 中的每个 Token 转换为 ID 。生成 Token IDs。
token_ids = tokenizer.convert_tokens_to_ids(token_sequence)
print(token_ids)
# tokenizer.encode 相当于上面 tokenizer.tokenize 与 tokenizer.convert_tokens_to_ids 的组合。
# token_ids = tokenizer.encode(text)  # type: ignore
# ================================================

# 使用 Token IDs 构造 Tensor(张量)。结果像这样：
# tensor([[151644,    872,    198,   6023,     11,    600,   2776,   3874,    380,
#           10159,  56191, 151645,    198, 151644,  77091,    198, 151667,    271,
#          151668,    271]])
# TODO: to 是干什么用的？为什么要移动设备？
input_ids = torch.tensor([token_ids]).to(model.device)
print(input_ids)
# 生成注意力掩码，用于表示模型关注关注的 Token。1 表示关注，0 表示不关注。
attention_mask = torch.ones_like(input_ids)
print(attention_mask)
# 将 input_ids 和 attention_mask 组装到到一个字典中。作为模型的输入。
inputs_ids_with_mask = {"input_ids": input_ids, "attention_mask": attention_mask}
print(inputs_ids_with_mask)
# ###################################################################################################################
# ！！！注意：attention_mask 参数并不是必须的！！！
# 可以用这种 inputs_ids = {"input_ids": input_ids} 直接传递给模型，只不过会出现下面这种警告：
# The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# 但是依然可以推理出结果。
# ！！！所以传入模型的内容本质就是 input_ids！！！
# 最终生成需要输入给模型的 input_ids，结果像下面这样：
# {'input_ids': tensor([[151644,    872,    198,   6023,     11,    600,   2776,   3874,    380,
#           10159,  56191, 151645,    198, 151644,  77091,    198, 151667,    271,
#          151668,    271]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
# ###################################################################################################################

# ===========================================================
# ================ 二、！！！模型进行推理！！！ ================
# ===========================================================
# 输入 Tensor，输出 Tensor。这是模型进行计算（i.e. 推理）的过程，也是整个代码，耗费时间最长的一步。
output_ids = model.generate(**inputs_ids_with_mask, max_new_tokens=32768)  # type: ignore
print(output_ids)

# ====================================================
# ================ 三、处理推理后的结果 ================
# ====================================================
#
# =========================================================================================================
# 下面这段代码，是一套简化的代码，decode() 方法只需传入特定参数，即可完整实现 “三、处理推理后的结果”
# =========================================================================================================
# assistant_reply = tokenizer.decode(
#     output_ids[0][len(inputs_ids_with_mask["input_ids"][0]) :],
#     skip_special_tokens=True,
# )
# print(assistant_reply)
# messages.append({"role": "assistant", "content": assistant_reply})  # type: ignore
# print(messages)
# =========================================================================================================
#
# 解构 Tensor 到 token ids
# output_token_ids = output_ids[0].tolist()
# 为了性能，通常会在这里就去掉用户输入，使用 `[len(input_ids[0]) :]` 只保留模型生成的部分返回给用户。
output_token_ids = output_ids[len(input_ids[0]) :]
print(output_token_ids)
# 去掉 Control Token
output_token_ids_filtered = []
for tid in output_token_ids:
    if tid not in tokenizer.all_special_ids:
        output_token_ids_filtered.append(tid)
print(output_token_ids_filtered)
# 将 output_token_ids 中的每个 Token 转换为 Token sequence。
output_token_sequence = tokenizer.convert_ids_to_tokens(output_token_ids_filtered)
print(output_token_sequence)
output_text = tokenizer.convert_tokens_to_string(output_token_sequence)  # type: ignore
print(output_text)

# ==================================================================================
# ================ 四、应用聊天模板，将模型生成的文本添加到 messages 中 ================
# ==================================================================================
messages.append({"role": "assistant", "content": output_text})
print(messages)
