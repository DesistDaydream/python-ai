from huggingface_hub import constants

# 查看当前生效的 HF 根目录
print(f"HF Home: {constants.HF_HOME}")

# 查看模型具体的缓存目录
print(f"Hub Cache: {constants.HF_HUB_CACHE}")
