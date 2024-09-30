import pandas as pd
import matplotlib.pyplot as plt


def visualize_word_level_from_pkl(pkl_file, line_number):
    df = pd.read_pickle(pkl_file)
    word_level_data = df.at[line_number, 'word_level']

    # 如果数据是字符串，解析JSON
    if isinstance(word_level_data, str):
        tokens_data = json.loads(word_level_data)
    else:
        tokens_data = word_level_data

    # 生成唯一标签
    tokens_df = pd.DataFrame(tokens_data['tokens'])
    tokens_df['unique_token'] = tokens_df['token'] + '_' + tokens_df.groupby('token').cumcount().astype(str)

    # 绘图
    plt.figure(figsize=(10, 10))
    plt.bar(tokens_df['unique_token'], tokens_df['value'], color='blue')
    plt.xlabel('Tokens')
    plt.ylabel('Values')
    plt.title('Token Values Visualization')
    plt.xticks(rotation=90)  # 可能需要旋转标签以提高可读性
    plt.show()


# 使用示例
# visualize_word_level_from_pkl('path_to_your_file.pkl', line_number)


# 使用示例
visualize_word_level_from_pkl('/Users/ximing/Desktop/Explainprompt/openai_logit/45250_45350perturbed_df.pkl', 29)
# 001 /Users/ximing/Desktop/Explainprompt/openai_logit/45010_45030perturbed_inferenced_df.pkl    4
# 002/Users/ximing/Desktop/Explainprompt/openai_logit/45010_45030perturbed_inferenced_df.pkl    5
# 003 /Users/ximing/Desktop/Explainprompt/openai_logit/45010_45030perturbed_inferenced_df.pkl    10
#004 '/Users/ximing/Desktop/Explainprompt/openai_logit/45250_45350perturbed_df.pkl 29
#/Users/ximing/Desktop/Explainprompt/openai_logit/45450_45550perturbed_inferenced_df.pkl 73