# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# def visualize_word_level_from_pkl(pkl_file, line_number):
#     df = pd.read_pickle(pkl_file)
#     word_level_data = df.at[line_number, 'word_level']
#
#     # 如果数据是字符串，解析JSON
#     if isinstance(word_level_data, str):
#         tokens_data = json.loads(word_level_data)
#     else:
#         tokens_data = word_level_data
#
#     # 生成唯一标签
#     tokens_df = pd.DataFrame(tokens_data['tokens'])
#     tokens_df['unique_token'] = tokens_df['token'] + '_' + tokens_df.groupby('token').cumcount().astype(str)
#
#     # 绘图
#     plt.figure(figsize=(10, 10))
#     plt.bar(tokens_df['unique_token'], tokens_df['value'], color='blue')
#     plt.xlabel('Tokens')
#     plt.ylabel('Values')
#     plt.title('Token Values Visualization')
#     plt.xticks(rotation=90)  # 可能需要旋转标签以提高可读性
#     plt.show()
#
#
# # 使用示例
# # visualize_word_level_from_pkl('path_to_your_file.pkl', line_number)
#
#
# # 使用示例
# visualize_word_level_from_pkl('/Users/ximing/Desktop/Explainprompt/openai_logit/45250_45350perturbed_df.pkl 2', 29)
# 001 /Users/ximing/Desktop/Explainprompt/openai_logit/45010_45030perturbed_inferenced_df.pkl    4
# 002/Users/ximing/Desktop/Explainprompt/openai_logit/45010_45030perturbed_inferenced_df.pkl    5
# 003 /Users/ximing/Desktop/Explainprompt/openai_logit/45010_45030perturbed_inferenced_df.pkl    10
#004 '/Users/ximing/Desktop/Explainprompt/openai_logit/45250_45350perturbed_df.pkl 29
#/Users/ximing/Desktop/Explainprompt/openai_logit/45450_45550perturbed_inferenced_df.pkl 73
import matplotlib.pyplot as plt

# Data from the table
K_values = [5, 10, 30, 50]
conf_top_ins = [1.0, 0.89, 0.91, 0.9]
conf_bot_ins = [0.44, 0.204, 0.21, 0.27]
conf_top_query = [0.08, 0.28, 0.25,0.26 ]
conf_bot_query = [0.07, 0.25, 0.17, 0.22]

eql_top_ins = [0.85, 0.87, 0.89, 0.87]
eql_bot_ins = [0.34, 0.33, 0.32, 0.29]
eql_top_query = [0.28, 0.31, 0.33,0.30 ]
eql_bot_query = [0.21, 0.26, 0.24, 0.26]

# Plotting the data
plt.figure(figsize=(6, 6))
plt.plot(K_values, conf_top_ins, marker='o', linestyle='-',label='Confidence Treatment Instruction')
plt.plot(K_values, conf_bot_ins, marker='o', linestyle='-',label='Confidence Control Instruction')
plt.plot(K_values, conf_top_query, marker='o', linestyle='-',label='Confidence Treatment Query')
plt.plot(K_values, conf_bot_query, marker='o', linestyle='-',label='ConfidenceControl Query')

plt.plot(K_values, eql_top_ins, marker='o', linestyle='--',label='Equal Treatment Instruction')
plt.plot(K_values, eql_bot_ins, marker='o', linestyle='--',label='Equal Control Instruction')
plt.plot(K_values, eql_top_query, marker='o', linestyle='--',label='Equal Treatment Query')
plt.plot(K_values, eql_bot_query, marker='o', linestyle='--',label='Equal Control Query')


plt.xlabel('Aggregation steps',fontsize=14)
plt.ylabel('Flip rate',fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
