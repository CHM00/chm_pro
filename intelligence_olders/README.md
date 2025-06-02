<<<<<<< HEAD
---
license: CC BY-NC 4.0
#用户自定义标签
tags:
  - Datawhale

text:
  #二级只能属于一个task_categories
  auto-speech-recognition:
    #三级可以多选
    languages:
      - cn

---  

# 语音识别与意图分析系统

本项目是一个基于语音识别和大语言模型的智能分析系统，可以从音频文件中识别语音内容，并分析用户的外卖点餐意图及食物信息。

## 功能特点

* 批量处理音频文件（支持 WAV、MP3 等格式）
* 使用 FunASR 进行高精度语音识别（ASR）
* 通过大语言模型（LLM）分析文本中的外卖意图
* 提取用户想要点的具体食物信息
* 支持高并发请求处理，提高效率
* 结果以结构化格式保存，便于后续处理

## 项目结构

```
.
├── project/
│   └── audio/             # 存放待处理的音频文件 在官方下载：main_test_organized.py README.md requirements.txt A.txt
├── output/                # 输出目录
│   └── output.txt         # 结果输出文件
├── main_test_organized.py # 主程序脚本
├── model.py               # FunASR模型定义文件（如果使用remote_code）
├── requirements.txt       # 项目依赖
├── .env                   # 环境变量配置（API密钥等）- 不要上传到公共仓库
├── README.md              # 本说明文件
└── A.txt                  # UUID顺序文件
```

## 安装步骤

1. **克隆仓库**（如果适用）：
   ```bash
   # git clone <仓库地址>
   # cd <项目目录>
   ```

2. **创建Python虚拟环境**（推荐）：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows系统使用: venv\Scripts\activate
   ```

3. **安装依赖**：
   * **PyTorch安装**：FunASR需要特定版本的PyTorch，建议根据您的系统配置（CPU/GPU，CUDA版本）选择合适的安装命令：
     ```bash
     # 对于CUDA 12.1
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     
     # 对于CUDA 11.8
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     
     # 仅CPU版本
     pip install torch torchvision torchaudio
     ```
     详情请参考[PyTorch官方安装指南](https://pytorch.org/get-started/locally/)
     
   * **其他依赖**：
     ```bash
     pip install -r requirements.txt
     ```

4. **创建环境变量文件**：
   创建一个名为`.env`的文件，添加以下内容：
   ```
   ARK_API_KEY="您的API密钥"
   ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"  # 或者您的特定端点
   ```
   **重要**：添加`.env`到您的`.gitignore`文件，避免敏感信息泄露。

5. **准备音频文件**：
   将您的音频文件（如`.wav`、`.mp3`等）放在`project/audio/`目录中。

6. **FunASR模型文件**：
   * 确保`main_test_organized.py`中指定的FunASR模型文件（`CONFIG["model_dir"] = "iic/SenseVoiceSmall"`）可访问。FunASR通常会在首次运行时自动下载。
   * 如果使用`remote_code="./model.py"`，请确保`model.py`文件存在于根目录，并包含模型所需的代码。

## 配置说明

您可以在`main.py`文件的`CONFIG`字典中调整以下设置：

* `audio_folder`：音频文件夹路径
* `model_dir`：ASR模型目录
* `max_files`：每次运行处理的最大文件数（用于测试或分批处理）
* `output_dir`：ASR输出目录
* `output_file`：结果文件
* `llm_model`：使用的LLM模型名称
* `llm_temperature`：LLM生成参数

## 运行方法

从项目根目录执行主脚本：

```bash
python main_test_organized.py
```

脚本将执行以下步骤：
1. 在指定文件夹中查找音频文件
2. 加载ASR模型
3. 对音频文件进行语音识别
4. 调用LLM分析识别文本
5. 将结果保存到`output.txt`文件中

结果文件格式为制表符分隔的文本文件，包含以下字段：
- 任务名称
- ASR识别文本
- 是否有外卖意图（1表示有，0表示无）
- 识别出的食物名称（如果有）

## 问题排查

* **PyTorch相关错误**：确保安装了正确版本的PyTorch，与您的系统配置兼容。
* **ASR模型下载问题**：首次运行时，FunASR可能需要下载模型文件，请确保网络连接良好。
* **API密钥错误**：检查`.env`文件中的API密钥是否正确。
* **音频文件无法识别**：确保音频文件格式受支持，质量良好，尝试不同的VAD参数。
* **LLM分析结果不准确**：可尝试调整提示词或模型参数。
=======
# 饿了么算法大赛-智慧养老比赛

## 比赛代码, 更新后
>>>>>>> 7f4aaca797298961f3e02f0d98a5fc9ba9f00c98
