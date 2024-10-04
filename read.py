import pickle
import tiktoken
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textstat

input_file = "hg3/5303_5403_discretize_llama3_qa_new_postprocess_inferenced_df.pkl"
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

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




def add_output(example):

    output_length = len(encoding.encode(example["real_output"]))
    return output_length
def readable(example):
    score = textstat.flesch_reading_ease(example["real_output"])
    return score

with open(input_file, "rb") as f:
    reconstructed_df = pickle.load(f)
    # columns_to_remove = ['query_max_normalized_value',
    #                      'query_max_token', 'instruction_max_normalized_value',
    #                      'instruction_max_token']
    # reconstructed_df = reconstructed_df.drop(columns=columns_to_remove, axis=1)
    reconstructed_df['real_output_length'] = reconstructed_df.apply(lambda row: add_output(row), axis=1)
    reconstructed_df['real_output_readable_score'] = reconstructed_df.apply(lambda row: readable(row), axis=1)
    new_columns = reconstructed_df.apply(process_row, axis=1)
    df = pd.concat([reconstructed_df, new_columns], axis=1)

df.to_pickle(input_file)

with open(input_file, "rb") as f:
    df1 = pickle.load(f)


with open("hg1/5303_5503_qa_hg_baseline_inferenced_df.pkl", "rb") as f:
    df2 = pickle.load(f)
    df2 = df2[:99]
    filtered_df2 = df2[df2['prompt'].isin(df1['query'])]
    filtered_df2['real_output_length'] = filtered_df2.apply(lambda row: add_output(row), axis=1)

# df1.set_index('id', inplace=True)
# df2.set_index('id', inplace=True)

# 执行相减操作
df1['baseline_real_output_readable_score'] = filtered_df2['real_output_length']
df1['difference'] = (df1['real_output_length'] - df1['baseline_real_output_readable_score'])

df1.to_pickle(input_file)
#1103_1203_logits_new_prompt_qa_postprocess_inferenced_df.pkl
with open(input_file, "rb") as f:
    reconstructed_df = pickle.load(f)
    # spearman_corr = reconstructed_df['instruction_weight'].corr(reconstructed_df['difference'], method='spearman')
    # print("Spearman Correlation coefficient:", spearman_corr)

    print(reconstructed_df.columns)
    #reconstructed_df = reconstructed_df[reconstructed_df['instruction_max_token'].notna()]

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

    # print(reconstructed_df.columns)

    plt.figure(figsize=(10, 10))  # 设置图形大小
    # x = np.linspace(40, 70, 100)  # 从0到10生成100个点
    # y = pearson * x  # 计算y值
    #
    # # 绘制y = 0.59x线
    # plt.plot(x, y, 'r-', label=f'y = {pearson}x')  # 'b-' 表示蓝色实线
    #
    # # 添加方程和相关系数注释
    # plt.text(5, 3, f'y = {pearson}x\nr = {pearson}', fontsize=12, color='blue')  # 调整位置和字体大小
    # if "similarity" in input_file:
    #     method = "Similarity"
    # elif "logits" in input_file:
    #     method = "Logits"
    # else:
    #     method = "Discretize"
    # filtered_df = reconstructed_df[
    #     (reconstructed_df['real_output_readable_score'] >= 0) &
    #     (reconstructed_df['baseline_real_output_readable_score'] >= 0)
    #     ]
    # plt.scatter(filtered_df['real_output_readable_score'], filtered_df['baseline_real_output_readable_score'], c='blue', marker='o')  # 绘制散点图
    # plt.title(f'{method}method 0-100')  # 添加标题
    # plt.xlabel('real_output_readable_score')  # 设置横轴标签
    # plt.ylabel('baseline_real_output_readable_score')  # 设置��轴标签
    #
    # plt.legend()  # 显示图例
    # plt.grid(True)  # 显示网格
    # plt.show()
#
#
# import pandas as pd
# import os
#
#
# def load_and_combine_pkl_files(directory_path):
#     # List to hold all the dataframes
#     dataframes = []
#
#     # Loop through all the files in the specified directory
#     for filename in os.listdir(directory_path):
#         if filename.endswith("df.pkl"):
#             # Construct full file path
#             file_path = os.path.join(directory_path, filename)
#             # Load the dataframe from a pkl file
#             df = pd.read_pickle(file_path)
#             # Append the dataframe to the list
#             dataframes.append(df)
#
#     # Concatenate all dataframes into one big dataframe
#     #big_df = pd.concat(dataframes, ignore_index=True)
#
#    # return big_df
#
#
# # Usage
# directory_path = '/Users/ximing/Desktop'
# big_df = load_and_combine_pkl_files(directory_path)
# big_df = big_df
# print(big_df)
