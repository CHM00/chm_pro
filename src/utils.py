import inspect

def function_to_json(func) -> dict:
    # 获取函数的参数信息
    sig = inspect.signature(func)
    parameters = {}
    required = []
    doc = inspect.getdoc(func) or ""
    param_descriptions = {}
    # 从函数文档中提取参数描述
    if doc:
        lines = doc.split('\n')
        for line in lines:
            if line.strip().startswith(':param'):
                parts = line.split(':', 3)
                if len(parts) >= 4:
                    param_name = parts[1].strip()
                    param_desc = parts[3].strip()
                    param_descriptions[param_name] = param_desc

    for param_name, param in sig.parameters.items():
        param_info = {
            "description": param_descriptions.get(param_name, ""),
        }
        # 根据参数注解推断参数类型
        if param.annotation is not param.empty:
            if param.annotation is int or param.annotation is float:
                param_info["type"] = "number"
            elif param.annotation is str:
                param_info["type"] = "string"
            elif param.annotation is bool:
                param_info["type"] = "boolean"
            else:
                param_info["type"] = "string"
        else:
            param_info["type"] = "string"

        # 处理参数默认值
        if param.default is not param.empty:
            param_info["default"] = param.default
        else:
            required.append(param_name)
        parameters[param_name] = param_info

    # 返回符合 OpenAI tool schema 的字典
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
