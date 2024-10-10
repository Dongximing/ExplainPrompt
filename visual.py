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
# visualize_word_level_from_pkl('/Users/ximing/Desktop/Explainprompt/openai_logit/45250_45350perturbed_df.pkl 29', 2)
# 001 /Users/ximing/Desktop/Explainprompt/openai_logit/45010_45030perturbed_inferenced_df.pkl    4
# 002/Users/ximing/Desktop/Explainprompt/openai_logit/45010_45030perturbed_inferenced_df.pkl    5
# 003 /Users/ximing/Desktop/Explainprompt/openai_logit/45010_45030perturbed_inferenced_df.pkl    10
#004 '/Users/ximing/Desktop/Explainprompt/openai_logit/45250_45350perturbed_df.pkl 29
#/Users/ximing/Desktop/Explainprompt/openai_logit/45450_45550perturbed_inferenced_df.pkl 73
import matplotlib.pyplot as plt

# Data from the table

# equal_treatment = [0.63,0.67,0.67,0.66,0.74]
# equal_control = [0.34,0.42,0.40,0.37,0.36]
# conf_treatment= [0.68,0.69,0.65,0.67]
# conf_control = [0.32,0.33,0.34,0.32]
#
# conf_treatment = [0.68,0.69,0.65,0.67]
# conf_control = [0.32,0.33,0.34,0.32]
# equal_treatment= [0.55,0.56,0.59,0.58]
# equal_control = [0.36,0.37,0.34,0.33]
# K_values = [10, 20, 30, 40]
# len_10 = [4.34,6.10,6.13,6.10]
# len_20 = [4.37,6.21,6.14,6.89]
# len_30 = [4.85,6.58,6.57,7.14]
# len_40 = [4.57,6.75,6.85,7.27]
# len_50 = [4.74,6.94,7.01,7.68]
#
#
#
#
# # Plotting the data
# plt.figure(figsize=(7, 7))
# # plt.plot(K_values, equal_treatment, marker='o', linestyle='-',label='Treatment')
# # plt.plot(K_values, equal_control, marker='o', linestyle='-',label='Control')
# plt.plot(K_values, len_10 , marker='o', linestyle='-',label='Input len = 10')
# plt.plot(K_values, len_20, marker='o', linestyle='',label='Input len = 20')
# plt.plot(K_values, len_30, marker='o', linestyle='-',label='Input len = 30')
# plt.plot(K_values, len_40, marker='o', linestyle='-',label='Input len = 40')
# plt.plot(K_values, len_50, marker='o', linestyle='-',label='Input len = 50')
#
# # plt.plot(K_values, eql_top_ins, marker='o', linestyle='--',label='Equal Treatment Instruction')
# # plt.plot(K_values, eql_bot_ins, marker='o', linestyle='--',label='Equal Control Instruction')
# # plt.plot(K_values, eql_top_query, marker='o', linestyle='--',label='Equal Treatment Query')
# # plt.plot(K_values, eql_bot_query, marker='o', linestyle='--',label='Equal Control Query')
#
# plt.xticks(fontsize=14)  # Increasing font size of the x-axis labels
# plt.yticks(fontsize=14)
# # Increase the font size of "Treatment" in the legend
#
# plt.xlabel('M- Aggregated rounds',fontsize=18)
# plt.ylabel('Flip rate',fontsize=18)
# plt.legend(fontsize=15)
#
# plt.grid(True)
# plt.show()

# import matplotlib.pyplot as plt
#

import matplotlib.pyplot as plt

# Data provided by the user
# input_lengths = [30, 60, 90, 120, 150]
# Agg_Equ = [3.95, 5.31, 6.77, 7.45, 9.79]
# Agg_Conf = [3.84, 5.23, 6.74, 7.36, 9.75]
# Inference = [2.05, 4.37, 5.39, 5.95, 7.07]
# plt.figure(figsize=(6, 6))
# # Creating the plot without specifying figure size
# plt.plot(input_lengths, Agg_Equ, marker='o', linestyle='-', label='Agg_Equ')
# plt.plot(input_lengths, Agg_Conf, marker='o', linestyle='-', label='Agg_Conf')
# plt.plot(input_lengths, Inference, marker='o', linestyle='-', label='Pure Inference')
#
# plt.yticks(range(0, 11))  # Setting y-axis ticks from 0 to 10
# plt.ylim(0, 10)
# plt.xlabel('Output Length', fontsize=18)
# plt.ylabel('Inference Time (seconds)', fontsize=18)
# plt.xticks(input_lengths)  # Ensuring all specified x-axis values are shown
# plt.legend(loc='lower right',fontsize=15)
# plt.grid(True)
# plt.show()




input_lengths = [10, 20, 30, 40, 50]
Pertb_Sim = [6.10, 6.89, 7.14, 7.27, 7.68]
Perb_Log = [6.10, 6.21, 6.58, 6.75, 6.94]
Perb_Dis = [6.13, 6.14, 6.57, 6.85, 7.01]
Inference = [4.34, 4.37, 4.85, 4.57, 4.74]
# input_lengths = [20, 320, 3200, 16000, 32000]
# Pertb_Sim = [0.63, 0.67,0.67,0.66,0.74]
# Perb_Log = [0.34,0.42,0.40,0.37,0.36]

plt.figure(figsize=(6, 6))
plt.plot(input_lengths, Pertb_Sim, marker='o', linestyle='-', label='Pertb_Sim')
plt.plot(input_lengths, Perb_Log, marker='o', linestyle='-', label='Perb_Log')
plt.plot(input_lengths, Perb_Dis, marker='o', linestyle='-', label='Perb_Dis')
plt.plot(input_lengths, Inference, marker='o', linestyle='-', label='Pure Inference')
#
# input_lengths = [30, 60, 90, 120, 150]
# Pertb_Sim = [6.10, 6.89, 7.14, 7.27, 7.68]
# Agg_Equ = [3.95, 5.31, 6.77, 7.45, 9.79]
# Agg_Conf = [3.84,5.23,6.74,7.36,9.75]
# Inference = [2.05, 4.37, 5.39, 5.95, 7.07]
#
# plt.figure(figsize=(6, 6))
# plt.plot(input_lengths, Agg_Equ, marker='o', linestyle='-', label='Agg_Equ')
# plt.plot(input_lengths, Agg_Conf, marker='o', linestyle='-', label='Agg_Conf')
# plt.plot(input_lengths, Agg_Conf, marker='o', linestyle='-', label='Agg_Conf')
#
# plt.plot(input_lengths, Inference, marker='o', linestyle='-', label='Pure Inference')
#

plt.yticks(range(0, 11))  # Setting y-axis ticks from 0 to 10
plt.ylim(0, 10)
plt.xlabel('Input Length',fontsize=18)
plt.ylabel('Inference Time (seconds)',fontsize=18)
plt.legend(loc='center right',fontsize=15)
plt.grid(True)
plt.show()

