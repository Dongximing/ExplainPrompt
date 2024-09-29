import pickle
import tiktoken
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
input_file = "1103_1203_discretize_new_prompt_qa_postprocess_inferenced_df.pkl"


def process_row(row):
    # 解析数据
    items = row['query_tokens']
    # 归一化并找到最大值及其token
    max_value = -1
    max_token = None

    for item in items:
        normalized_value = item['value']
        if normalized_value > max_value:
            max_value = normalized_value
            max_token = item['token']

    instruction_items = row['instructions_tokens']
    # 归一化并找到最大值及其token
    instruction_max_value = -1
    instruction_max_token = None
    for item in instruction_items:
        normalized_value = item['value']
        if normalized_value > instruction_max_value:
            instruction_max_value = normalized_value
            instruction_max_token = item['token']



    return pd.Series([max_value, max_token,instruction_max_value,instruction_max_token], index=['query_max_normalized_value', 'query_max_token', 'instruction_max_normalized_value', 'instruction_max_token'])







encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def add_output(example):
    # print(f"Original output: {example['real_output']}")
    # print(type(example['real_output']))
    output_length = len(encoding.encode(example["real_output"]))
    return output_length
with open(input_file, "rb") as f:
    reconstructed_df = pickle.load(f)
    # columns_to_remove = ['query_max_normalized_value',
    #                      'query_max_token', 'instruction_max_normalized_value',
    #                      'instruction_max_token']
    # reconstructed_df = reconstructed_df.drop(columns=columns_to_remove, axis=1)
    reconstructed_df['real_output_length'] = reconstructed_df.apply(lambda row: add_output(row), axis=1)
    new_columns = reconstructed_df.apply(process_row, axis=1)
    df = pd.concat([reconstructed_df, new_columns], axis=1)

df.to_pickle(input_file)

with open(input_file, "rb") as f:
    df1 = pickle.load(f)


with open("1103_1203_baseline_qa_inferenced_df.pkl", "rb") as f:
    df2 = pickle.load(f)
# df1.set_index('id', inplace=True)
# df2.set_index('id', inplace=True)

# 执行相减操作
df1['difference'] = (df1['real_output_length'] - df2['input_cost'])
df1['baseline'] = df2['input_cost']
df1.to_pickle(input_file)
#1103_1203_logits_new_prompt_qa_postprocess_inferenced_df.pkl
with open(input_file, "rb") as f:
    reconstructed_df = pickle.load(f)
    print(reconstructed_df.columns)
    # 计算两列之间的皮尔逊相关系数
    spearman_corr = reconstructed_df['query_max_normalized_value'].corr(reconstructed_df['difference'],method='pearson'
                                                                        )

    print("Pearson correlation coefficient:", spearman_corr)

    pearson = reconstructed_df['instruction_max_normalized_value'].corr(reconstructed_df['difference'], method='pearson')
    print("pearson Correlation coefficient: instruction_max_normalized_value", pearson)


    spearman_corr = reconstructed_df['query_max_normalized_value'].corr(reconstructed_df['difference'], method='spearman')
    print("Spearman Correlation coefficient: query_max_normalized_value", spearman_corr)
    pearson = reconstructed_df['instruction_max_normalized_value'].corr(reconstructed_df['difference'], method='spearman')
    print("Spearman Correlation coefficient:instruction_max_normalized_value ", pearson)

    plt.figure(figsize=(10, 10))  # 设置图形大小
    plt.scatter(reconstructed_df['max_normalized_value'], reconstructed_df['difference'], c='blue', marker='o')  # 绘制散点图
    plt.title('discretize method 0-100')  # 添加标题
    plt.xlabel('max_normalized_value')  # 设置横轴标签
    plt.ylabel('difference')  # 设置��轴标签


    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()


