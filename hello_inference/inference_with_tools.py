from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedTokenizerBase
from transformers import BatchEncoding

# model_name = "Qwen/Qwen3-0.6B"
model_name = r"D:\appdata\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
print(tokenizer)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# 工具声明
tools: list = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

prompt = "北京今天天气如何"
messages = [
    {"role": "user", "content": prompt},
]


model_inputs: BatchEncoding = tokenizer.apply_chat_template(
    messages,
    # 将可用的工具添加到模型输入中
    tools=tools,
    tokenize=True,
    return_tensors="pt",
    add_generation_prompt=False,
    enable_thinking=True,
).to(model.device)  # type: ignore

output_ids = model.generate(**model_inputs, max_new_tokens=32768)  # type: ignore
print(output_ids)

# 解码的参数不使用 output_ids[0][len(model_inputs["input_ids"][0]) :]，因为要检查 tool_call 是否存在
assistant_reply = tokenizer.decode(
    output_ids[0][len(model_inputs["input_ids"][0]) :],  # type: ignore
    skip_special_tokens=True,
)
print(assistant_reply)


# 若 LLM 判断需要使用工具，将会返回 tool_call 标签包裹的内容。输出效果类似下面这样：
# <tool_call>
# {"name": "get_weather", "arguments": {"city": "北京"}}
# </tool_call>
