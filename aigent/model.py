from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-XbQUrVlV4RurIplE3958B596Bb70404b9eC7F7Ef4b401804",  # 替换为你的 API Key
    base_url="https://aihubmix.com/v1", # 使用 SiliconFlow 的 API 地址
)

# 指定模型名称
model_name = "Qwen/Qwen3-8B"