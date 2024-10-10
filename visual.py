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
# equal_treatment = [0.63,0.67,0.67,0.66,0.74]
# equal_control = [0.34,0.42,0.40,0.37,0.36]
# conf_treatment= [0.68,0.69,0.65,0.67]
# conf_control = [0.32,0.33,0.34,0.32]

conf_treatment = [0.68,0.69,0.65,0.67]
conf_control = [0.32,0.33,0.34,0.32]
equal_treatment= [0.55,0.56,0.59,0.58]
equal_control = [0.36,0.37,0.34,0.33]



# Plotting the data
plt.figure(figsize=(7, 7))
# plt.plot(K_values, equal_treatment, marker='o', linestyle='-',label='Treatment')
# plt.plot(K_values, equal_control, marker='o', linestyle='-',label='Control')
plt.plot(K_values, equal_treatment, marker='o', linestyle='--',label='Treatment (Agg_Equ)')
plt.plot(K_values, equal_control, marker='o', linestyle='--',label='Control (Agg_Equ)')
plt.plot(K_values, conf_treatment, marker='o', linestyle='-',label='Treatment (Agg_Conf)')
plt.plot(K_values, conf_control, marker='o', linestyle='-',label='Control (Agg_Conf)')

# plt.plot(K_values, eql_top_ins, marker='o', linestyle='--',label='Equal Treatment Instruction')
# plt.plot(K_values, eql_bot_ins, marker='o', linestyle='--',label='Equal Control Instruction')
# plt.plot(K_values, eql_top_query, marker='o', linestyle='--',label='Equal Treatment Query')
# plt.plot(K_values, eql_bot_query, marker='o', linestyle='--',label='Equal Control Query')

plt.xticks(fontsize=14)  # Increasing font size of the x-axis labels
plt.yticks(fontsize=14)
# Increase the font size of "Treatment" in the legend

plt.xlabel('M- Aggregated rounds',fontsize=18)
plt.ylabel('Flip rate',fontsize=18)
plt.legend(fontsize=15)

plt.grid(True)
plt.show()
