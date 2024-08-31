from datetime import datetime

import torch
import transformers
from transformers import AutoTokenizer

model_id = "/Users/zaoyingouyuandibaozha/Documents/model/Meta-Llama-3.1-8B-Instruct"

start_time = datetime.now()

# 加载模型

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",  # auto （device_map 确保模型被移动到您的GPU(s)上，如果有的话。）
)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",  # auto （device_map 确保模型被移动到您的GPU(s)上，如果有的话。）
# )

# 使用 AutoTokenizer 加载一个分词器
# 预处理文本输入
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# 调用generate()方法来返回生成的token
outputs = pipeline(
    prompt,
    max_new_tokens=1024,  # 返回的最大新tokens数量
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,  # Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
)

# 输出 转换为文本
print(outputs[0]["generated_text"][len(prompt):])

# q1 = "Who are you?"
# Nice to meet you! I am LLaMA, a helpful assistant developed by Meta AI that can understand and respond to human input in a conversational manner. I'm here to assist you with any questions, tasks, or topics you'd like to discuss. I can provide information on a wide range of subjects, from science and history to entertainment and culture. I can also help with language-related tasks such as language translation, text summarization, and even creative writing. My goal is to be a helpful and informative companion, so feel free to ask me anything!
# total spend: 254 seconds
