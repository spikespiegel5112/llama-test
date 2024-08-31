from datetime import datetime

import torch
import transformers
from flask import Blueprint, Flask, request
from transformers import AutoTokenizer

app = Flask(__name__)

model_id = "/Users/zaoyingouyuandibaozha/Documents/model/Meta-Llama-3.1-70B-Instruct"

llama_controller = Blueprint('LlamaController', __name__)
pipeline = None


@llama_controller.route('/helloWorld', methods=['GET'])
def hello_world():
    print('sdssasd')
    return "<p>Hello, World!</p>"


def load_model():
    global pipeline
    start_time = datetime.now()
    print('start time:')
    print(start_time)
    # 加载模型
    if pipeline is None:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="mps",  # auto （device_map 确保模型被移动到您的GPU(s)上，如果有的话。）
        )

    return pipeline


@llama_controller.route('/executePipline', methods=['POST'])
def execute_pipline():
    pipeline = load_model()
    print('sdssasd')

    # 使用 AutoTokenizer 加载一个分词器
    # 预处理文本输入
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    message = request.json['message']
    init_message = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        init_message + message,
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
    print(message)

    result = outputs[0]["generated_text"][len(prompt):]
    context = message + [{"role": "assistant", "content": result}]
    # 输出 转换为文本
    print(result)

    # q1 = "Who are you?"
    # Nice to meet you! I am LLaMA, a helpful assistant developed by Meta AI that can understand and respond to human input in a conversational manner. I'm here to assist you with any questions, tasks, or topics you'd like to discuss. I can provide information on a wide range of subjects, from science and history to entertainment and culture. I can also help with language-related tasks such as language translation, text summarization, and even creative writing. My goal is to be a helpful and informative companion, so feel free to ask me anything!
    # total spend: 254 seconds

    end_time = datetime.now()
    print('end time:')
    print(end_time)
    return context
