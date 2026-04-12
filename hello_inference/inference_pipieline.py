from transformers import pipeline, TextGenerationPipeline
from transformers import GenerationConfig

# https://huggingface.co/Qwen/Qwen3-0.6B
# model_name = "Qwen/Qwen3-0.6B"
# model_path = r"D:\appdata\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
# model_path = r"D:\appdata\models\desistdaydream"
model_path = "/mnt/d/appdata/models/desistdaydream/"

prompt = "hi, i'm DesistDaydream"
messages = [
    {"role": "user", "content": prompt},
]

# https://huggingface.co/docs/transformers/main/en/main_classes/pipelines
# https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.TextGenerationPipeline
# pipeline 是使用模型进行推理的一种便捷有效的方式。
# Pipeline 对象抽象了大部分细节，用户只需要关注模型的输入输出即可
# Pipeline 对象实现了类似 Go, Java 的 interface 效果。根据传入参数的不同，有非常多的实现。
# 比如 task="text-generation" 将会返回一个实现了 Pipeline 的 TextGenerationPipeline 对象
generator: TextGenerationPipeline = pipeline(
    task="text-generation",
    model=model_path,
    dtype="auto",
    device_map="auto",
)

# https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
# 模型的配置文件 generation_config.json 中有默认的参数值，这里可以覆盖默认值
gen_config = GenerationConfig(
    max_new_tokens=512,
    temperature=0.7,
)
response = generator(messages, generation_config=gen_config)
print("问题: {}\n回答: {}".format(prompt, response[0]["generated_text"][-1]["content"]))
