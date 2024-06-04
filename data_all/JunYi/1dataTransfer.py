"""
generate assist09-like data format for junyi
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


logs_file = './junyi_ProblemLog_original.csv'
ques_file = "./junyi_Exercise_table.csv"
save_file = './Junyi/junyi.csv'

ques_df = pd.read_csv(ques_file)[['name', 'topic', 'area']]
ques_df.rename(columns={'name': 'exercise'}, inplace=True)
logs_df = pd.read_csv(logs_file)

user_list = list(logs_df['user_id'])
ques_list = list(ques_df['exercise'])


# print("num of question in ques_df:%d" % len(set(ques_df['exercise'])))
# print("num of question in logs_df:%d" % len(set(logs_df['exercise'])))
# df = pd.merge(logs_df, ques_df, how='inner', on=['exercise'])
# print("num of question in df:%d" % len(set(df['exercise'])))

# df['correct'] = df['correct'].apply(lambda x: int(x))
# df.sort_values(by=['user_id'], ascending=True, inplace=True)
# print(df['correct'])
# df.to_csv(save_file)
# print(df.head())
