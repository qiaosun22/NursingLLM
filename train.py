# 作者：亚东
# 链接：https://zhuanlan.zhihu.com/p/700844670
# 来源：知乎
# 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

import torch
# from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
import json
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType
import torch
# import deepspeed
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct', cache_dir='/root/autodl-tmp', revision='master')

# def process_func(example):
#     MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
#     input_ids, attention_mask, labels = [], [], []
#     instruction = tokenizer(f"<|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
#     response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
#     input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
#     attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
#     labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
#     if len(input_ids) > MAX_LENGTH:  # 做一个截断
#         input_ids = input_ids[:MAX_LENGTH]
#         attention_mask = attention_mask[:MAX_LENGTH]
#         labels = labels[:MAX_LENGTH]
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels
#     }
    

# tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', use_fast=False, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("shenzhi-wang/Llama3-8B-Chinese-Chat", use_fast=False, trust_remote_code=True)
# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', device_map="auto",torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("shenzhi-wang/Llama3-8B-Chinese-Chat", torch_dtype=torch.bfloat16)



# 加载自定义格式的JSON文件
with open('custom_conversations.json', 'r') as f:
    data = json.load(f)

# 自定义数据处理函数
def process_conversations(data):
    conversations = []
    for conv in data:
        for turn in conv['chat']:
            conversations.append({
                "conversation_id": conv["id"],
                "turn_id": turn["id"],
                "user": turn["user_msg"],
                "assistant": turn["bot_response"]
            })
    return conversations

# 应用数据处理函数
processed_data = process_conversations(data)

# 创建Dataset对象
dataset = Dataset.from_pandas(pd.DataFrame(processed_data))

# # 选择预训练模型的分词器
# tokenizer = AutoTokenizer.from_pretrained("shenzhi-wang/Llama3-8B-Chinese-Chat")

# 定义分词函数
def tokenize_function(examples):
    model_inputs = tokenizer(examples['user'], examples['assistant'], padding="max_length", truncation=True, max_length=100)
    # Create labels
    labels = model_inputs['input_ids'].copy()
    model_inputs['labels'] = labels
    return model_inputs

# 对数据集进行分词和编码
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 将数据集拆分为训练集和验证集
split_dataset = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# 使用DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)




config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)

args = TrainingArguments(
    output_dir="./output/llama3",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=False,
    gradient_checkpointing=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=data_collator
)
trainer.train()

# lora_path='./llama3_lora'
# trainer.model.save_pretrained(lora_path)
# tokenizer.save_pretrained(lora_path)

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# from peft import PeftModel

# # mode_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct'
# lora_path = './llama3_lora' # lora权重路径

# # 加载tokenizer
# tokenizer = AutoTokenizer.from_pretrained(mode_path)

# # 加载模型
# model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16)

# # 加载lora权重
# model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

# prompt = "你是谁？"
# messages = [
#     # {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
#     {"role": "user", "content": prompt}
# ]

# text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

# generated_ids = model.generate(
#     model_inputs.input_ids,
#     max_new_tokens=512,
#     eos_token_id=tokenizer.encode('<|eot_id|>')[0]
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(response)


""""""

# import json
# import os
# import pandas as pd
# from datasets import Dataset, DatasetDict
# from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, DataCollatorForLanguageModeling
# import torch
# import deepspeed

# # Set a different port for distributed training
# os.environ['MASTER_PORT'] = '29501'  # Change this to an available port number

# # 加载自定义格式的JSON文件
# with open('custom_conversations.json', 'r') as f:
#     data = json.load(f)

# # 自定义数据处理函数
# def process_conversations(data):
#     conversations = []
#     for conv in data:
#         for turn in conv['chat']:
#             conversations.append({
#                 "conversation_id": conv["id"],
#                 "turn_id": turn["id"],
#                 "user": turn["user_msg"],
#                 "assistant": turn["bot_response"]
#             })
#     return conversations

# # 应用数据处理函数
# processed_data = process_conversations(data)

# # 创建Dataset对象
# dataset = Dataset.from_pandas(pd.DataFrame(processed_data))

# # 选择预训练模型的分词器
# tokenizer = AutoTokenizer.from_pretrained("shenzhi-wang/Llama3-8B-Chinese-Chat")

# # 定义分词函数
# def tokenize_function(examples):
#     model_inputs = tokenizer(examples['user'], examples['assistant'], padding="max_length", truncation=True, max_length=100)
#     # Create labels
#     labels = model_inputs['input_ids'].copy()
#     model_inputs['labels'] = labels
#     return model_inputs

# # 对数据集进行分词和编码
# tokenized_datasets = dataset.map(tokenize_function, batched=True)

# # 将数据集拆分为训练集和验证集
# split_dataset = tokenized_datasets.train_test_split(test_size=0.2)
# train_dataset = split_dataset['train']
# eval_dataset = split_dataset['test']

# # 使用DataCollatorForLanguageModeling
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # 选择预训练模型
# model = AutoModelForCausalLM.from_pretrained("shenzhi-wang/Llama3-8B-Chinese-Chat")

# # 设置训练参数
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     num_train_epochs=3,
#     weight_decay=3e-7,
#     save_total_limit=1,
#     fp16=True,  # 启用半精度训练
#     dataloader_num_workers=4,
#     deepspeed="ds_config.json",  # 添加DeepSpeed配置文件进行分布式训练
# )

# # 定义Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     data_collator=data_collator,
#     tokenizer=tokenizer,
# )

# # 初始化 DeepSpeed
# deepspeed.init_distributed()

# # 开始训练
# trainer.train()

# # 保存模型
# model.save_pretrained("./fine_tuned_llama3")
# tokenizer.save_pretrained("./fine_tuned_llama3")







# # import json
# # from datasets import Dataset, DatasetDict
# # import pandas as pd
# # from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, DataCollatorForLanguageModeling
# # import torch
# # import os


# # # Set a different port for distributed training
# # os.environ['MASTER_PORT'] = '29501'  # Change this to an available port number


# # # 加载自定义格式的JSON文件
# # with open('custom_conversations.json', 'r') as f:
# #     data = json.load(f)

# # # 自定义数据处理函数
# # def process_conversations(data):
# #     conversations = []
# #     for conv in data:
# #         for turn in conv['chat']:
# #             conversations.append({
# #                 "conversation_id": conv["id"],
# #                 "turn_id": turn["id"],
# #                 "user": turn["user_msg"],
# #                 "assistant": turn["bot_response"]
# #             })
# #     return conversations

# # # 应用数据处理函数
# # processed_data = process_conversations(data)

# # # 创建Dataset对象
# # dataset = Dataset.from_pandas(pd.DataFrame(processed_data))

# # # 选择预训练模型的分词器
# # tokenizer = AutoTokenizer.from_pretrained("shenzhi-wang/Llama3-8B-Chinese-Chat")

# # # 定义分词函数
# # def tokenize_function(examples):
# #     model_inputs = tokenizer(examples['user'], examples['assistant'], padding="max_length", truncation=True, max_length=100)
# #     # Create labels
# #     labels = model_inputs['input_ids'].copy()
# #     model_inputs['labels'] = labels
# #     return model_inputs

# # # 对数据集进行分词和编码
# # tokenized_datasets = dataset.map(tokenize_function, batched=True)

# # # 将数据集拆分为训练集和验证集
# # split_dataset = tokenized_datasets.train_test_split(test_size=0.2)
# # train_dataset = split_dataset['train']
# # eval_dataset = split_dataset['test']

# # # 使用DataCollatorForLanguageModeling
# # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # # 将模型移动到GPU（如果可用）
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # 选择预训练模型
# # model = AutoModelForCausalLM.from_pretrained("shenzhi-wang/Llama3-8B-Chinese-Chat").to(device).half()

# # # 设置训练参数
# # training_args = TrainingArguments(
# #     output_dir="./results",
# #     evaluation_strategy="epoch",
# #     learning_rate=2e-5,
# #     per_device_train_batch_size=1,
# #     per_device_eval_batch_size=1,
# #     num_train_epochs=3,
# #     weight_decay=3e-7,
# #     save_total_limit=1,
# #     fp16=True,  # 启用半精度训练
# #     dataloader_num_workers=4,
# #     deepspeed="ds_config.json",  # 添加DeepSpeed配置文件进行分布式训练
# # )

# # # 定义Trainer
# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=train_dataset,
# #     eval_dataset=eval_dataset,
# #     data_collator=data_collator,
# #     tokenizer=tokenizer,
# # )

# # # 开始训练
# # trainer.train()

# # # 保存模型
# # model.save_pretrained("./fine_tuned_llama3")
# # tokenizer.save_pretrained("./fine_tuned_llama3")
