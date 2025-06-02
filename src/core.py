# src/core.py (部分)
from openai import OpenAI
import json
from typing import List, Dict, Any
from utils import function_to_json
# 导入定义好的工具函数
from tools import get_current_datetime, add, compare, count_letter_in_string

SYSREM_PROMPT = """
你是一个叫不要葱姜蒜的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""


class Agent:
    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen2.5-32B-Instruct", tools: List = [],
                 verbose: bool = True):
        self.client = client
        self.tools = tools  # 存储可用的工具函数列表
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSREM_PROMPT},
        ]
        self.verbose = verbose

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # 使用 utils.function_to_json 获取所有工具的 JSON Schema
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # 处理来自模型的工具调用请求
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        # 动态执行工具函数
        # 注意：实际应用中应添加更严格的安全检查
        function_call_content = eval(f"{function_name}(**{function_args})")

        # 返回工具执行结果给模型
        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:
        # 主对话逻辑
        self.messages.append({"role": "user", "content": prompt})

        tool_schema = self.get_tool_schema()
        print(tool_schema)  # 打印工具 Schema 进行检查

        # 第一次调用模型，传入工具 Schema
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )

        # 检查模型是否请求调用工具
        if response.choices[0].message.tool_calls:
            tool_list = []
            # 处理所有工具调用请求
            for tool_call in response.choices[0].message.tool_calls:
                # 执行工具并将结果添加到消息历史中
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append(tool_call.function.name)
            if self.verbose:
                print("调用工具：", tool_list)

            # 第二次调用模型，传入工具执行结果
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),  # 再次传入 Schema 可能有助于模型理解上下文
                stream=False,
            )
            print(response)

        # 将最终的助手回复添加到消息历史
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content


# demo.py (部分)
if __name__ == "__main__":
    client = OpenAI(
        api_key="sk-XbQUrVlV4RurIplE3958B596Bb70404b9eC7F7Ef4b401804", # 替换为你的 API Key
        base_url="https://aihubmix.com/v1",
    )

    # 创建 Agent 实例，传入 client、模型名称和工具函数列表
    agent = Agent(
        client=client,
        model="Qwen/Qwen2.5-32B-Instruct",
        tools=[get_current_datetime, add, compare, count_letter_in_string],
        verbose=True # 设置为 True 可以看到工具调用信息
    )

    # 开始交互式对话循环
    print("开始交互式对话循环")
    while True:
        # 使用彩色输出区分用户输入和AI回答
        prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示
        if prompt.lower() == "exit":
            break
        response = agent.get_completion(prompt)
        print("\033[92mAssistant: \033[0m", response)  # 绿色显示AI助手回答
