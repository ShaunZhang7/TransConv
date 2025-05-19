import pandas as pd
import random
import csv
import pickle
import numpy as np

from rdkit import Chem
from rdkit.Chem import BRICS

from rdkit.Chem import Recap

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np



import math

x_values = [8.825034142, 6.095625401, 5.306943893, 4.297938347]
results = [10 ** (9 - x) for x in x_values]
print(results)

# 读取 CSV 文件
#df = pd.read_csv('C:\\Users\\zhao\\Desktop\\predict_with_results.csv')
# 生成新列
#df['Ki(nm)'] = 10 ** (9 - df['predict'])
# 保存结果到新文件
#df.to_csv('C:\\Users\\zhao\\Desktop\\predict_with_results1.csv', index=False)


# 生成分子指纹 ‌:ml-citation{ref="6" data="citationList"}
'''
def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2)

df = pd.read_csv('E:\\Shaun\\MolTrans-DTA\\dataset\\Kd\\train_origin.csv', index_col=0)
df['fp'] = df['SMILES'].apply(get_fingerprint)
# 指纹向量化并进行K-means聚类
X = np.array([np.frombuffer(fp.ToBitString().encode(), 'u1') for fp in df.fp])
kmeans = KMeans(n_clusters=10).fit(X)
df['cluster'] = kmeans.labels_
# 按聚类标签分层抽样
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['cluster'])
# 清理中间列
train_df.drop(['fp', 'cluster'], axis=1).to_csv('E:\\Shaun\\MolTrans-DTA\\dataset\\Kd\\train1.csv', index=False)
val_df.drop(['fp', 'cluster'], axis=1).to_csv('E:\\Shaun\\MolTrans-DTA\\dataset\\Kd\\val1.csv', index=False)
'''

# 散点图
'''import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
data = pd.read_csv('E:\\Shaun\\MolTrans-DTA\\Test\\predict_with_results1.csv')

# 提取实际值和预测值
actual_values = data['Label']
predicted_values = data['predict']

# 创建散点图
plt.figure(figsize=(10, 8)) #数据点密集在左下角，表明大部分预测值较低，实际值也较低，
# 但未提及具体颜色和大小要求，这里使用默认设置，即alpha=0.3来控制点的透明度
#plt.scatter(predicted_values, actual_values, alpha=1)
plt.scatter(
    predicted_values,
    actual_values,
    alpha=1,          # 透明度保持1（不透明）
    c='darkblue',         # 设置颜色为海军蓝
    s=10              # 设置点的大小（默认20，这里缩小为10）
)
# 设置坐标轴上下限，由于未给出具体范围，这里使用默认值，
# Matplotlib会根据数据自动调整显示范围，但可以通过plt.xlim()和plt.ylim()手动设置
plt.xlim(4, 11)  # 设置x轴范围为0到15
plt.ylim(4, 11)  # 设置y轴范围为0到15

# 添加标题和标签
plt.title('Scatter Plot of Actual vs Predicted Values')
plt.xlabel('Prediction')
plt.ylabel('Actual Value')

# 显示网格
plt.grid(True, alpha=0.5)

# 显示图像
plt.show()'''

# SMILES检测
'''
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
from tqdm import tqdm  # 可选：用于显示进度条

def check_smiles_validity(smiles):
    """
    检查SMILES的价键有效性
    返回值：
    - 'valid'：价键正确且可解析
    - 'valence_error'：价键错误
    - 'invalid'：无法解析的SMILES格式
    """
    # 尝试解析SMILES（不进行自动价键检查）
    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    if mol is None:
        return 'invalid'  # 无法解析的基础格式错误

    try:
        # 显式执行价键检查（使用 SANITIZE_PROPERTIES）
        Chem.SanitizeMol(mol, sanitizeOps=SanitizeFlags.SANITIZE_PROPERTIES)
        return 'valid'
    except ValueError as e:
        if 'Valence' in str(e):
            return 'valence_error'
        else:
            return 'invalid'  # 其他化学规则错误


# 读取CSV文件
input_file = 'E:\\Shaun\\MolTrans-master\\dataset\\BindingDB\\train(Miss95%)1.csv'
df = pd.read_csv(input_file)

# 检查每个SMILES（使用tqdm显示进度条）
tqdm.pandas(desc="Processing SMILES")
df['validation_status'] = df['SMILES'].progress_apply(check_smiles_validity)

# 分割数据
valid_df = df[df['validation_status'] == 'valid'].drop(columns=['validation_status'])
invalid_df = df[df['validation_status'] != 'valid'].drop(columns=['validation_status'])

# 保存结果
valid_file = 'E:\\Shaun\\MolTrans-master\\dataset\\DAVIS\\train_valid.csv'
invalid_file = 'E:\\Shaun\\MolTrans-master\\dataset\\DAVIS\\train_invalid.csv'
#valid_df.to_csv(valid_file, index=False)
#invalid_df.to_csv(invalid_file, index=False)

print(f"""
处理完成！
有效SMILES已保存至：{valid_file}（共{len(valid_df)}条）
无效SMILES已保存至：{invalid_file}（共{len(invalid_df)}条，包含：\n
 - 价键错误：{len(df[df['validation_status'] == 'valence_error'])}条
 - 格式错误：{len(df[df['validation_status'] == 'invalid'])}条
""")

#mols = [Chem.MolFromSmiles(s) for s in invalid_df['SMILES']] #  if pd.notnull(s)
#mols = [m for m in mols if m is not None]
# 批量分解分子
#hierarchies = [Recap.RecapDecompose(mol) for mol in mols]
#for i in new:
    #new.UpdatePropertyCache(strict=False)
#纠正不正确的化学键
#Chem.MolToSmiles(new[1], True)
'''

#Scare Data折线图
'''import matplotlib.pyplot as plt
import numpy as np

# 数据配置
models = {
    "TransConv":    [0.861, 0.847, 0.815, 0.771],
    "MolTrans":     [0.855, 0.835, 0.803, 0.769],
    "DeepDTI":      [0.852, 0.830, 0.769, 0.660],
    "DeepConv-DTI": [0.845, 0.824, 0.792, 0.725],
    "DeepDTA":      [0.839, 0.819, 0.788, 0.762]
}
x_labels = ['70%', '80%', '90%', '95%']
x = np.arange(len(x_labels))

# 创建画布
plt.figure(figsize=(10, 6), dpi=100)

# 绘制折线
for model, values in models.items():
    plt.plot(x, values, linestyle='-', linewidth=2, label=model)

# 图表装饰
plt.title("AUC-ROC Performance with Different Missing Data Ratios", fontsize=14, pad=20)
plt.xlabel("Percentage of Missing Dataset", fontsize=12)
plt.ylabel("AUC-ROC", fontsize=12)
plt.xticks(x, x_labels)
plt.ylim(0.73, 0.87)

# 移除网格线和调整图例位置
plt.grid(False)  # 关闭网格线
plt.legend(
    loc='lower left',          # 定位在左下角
    bbox_to_anchor=(0.02, 0.02),  # 微调位置（距离左/下边距2%）
    frameon=True,              # 显示图例外框
    fontsize=10                # 调小字体
)

plt.tight_layout()
plt.show()
# 保存图片（取消注释使用）
# plt.savefig('dti_performance.png', bbox_inches='tight')'''

# 读取CSV文件统计DT对信息
'''df = pd.read_csv('E:\\Shaun\\MolTrans-master\\dataset\\BIOSNAP_Add\\all.csv')
# 基础统计
stats = {
    "# Drugs": df['SMILES'].nunique(),                # 唯一SMILES数量‌:ml-citation{ref="1,6" data="citationList"}
    "# Proteins": df['Target Sequence'].nunique(),    # 唯一靶标序列数量‌:ml-citation{ref="1,5" data="citationList"}
    "# Pos Interactions": (df['Label'] == 1).sum(),   # 正样本数‌:ml-citation{ref="1,4" data="citationList"}
    "# Neg Interactions": (df['Label'] == 0).sum()    # 负样本数‌:ml-citation{ref="4" data="citationList"}
}
# 输出结果
print(pd.DataFrame.from_dict(stats, orient='index', columns=['Count']))'''

#Case Study预测结果筛选
'''# 读取原始CSV文件
df = pd.read_csv('D:\\Shaun\\MolTrans-DTA\\dataset\\Test\\predict_with_results.csv')
# 计算Label和predict两列值的差的绝对值
df['difference'] = abs(df['Label'] - df['predict'])
# 筛选出差的绝对值小于或等于0.2的行
filtered_df = df[df['difference'] <= 0.1]
# 删除不再需要的'difference'列
filtered_df = filtered_df.drop(columns=['difference'])
# 将筛选后的数据保存到一个新的CSV文件中
filtered_df.to_csv('D:\\Shaun\\MolTrans-DTA\\dataset\\Test\\predict_with_results_filter.csv', index=False)'''

# 更改列名(针对大型数据集)
'''df = pd.read_csv('D:\\Desktop\\DTI药物靶标亲和力预测\\TEFDTA-master\\data\\BindingDB\\BindingDB_train.csv')
df.rename(columns={
    'iso_smiles': 'SMILES',
    'target_sequence': 'Target Sequence',
    'affinity': 'Label'
}, inplace=True)
# 将修改后的DataFrame写回CSV文件
df.to_csv('D:\\Desktop\\DTI药物靶标亲和力预测\\TEFDTA-master\\data\\BindingDB\\train.csv', index=False)
'''

# 保存完全相同的，其它的也保存
'''
file1 = pd.read_csv('D:\\Shaun\\MolTrans-master\\dataset\\DAVIS\\all.csv')  # 替换为文件1的路径
file2 = pd.read_csv('D:\\Shaun\\MolTrans-master\\dataset\\BIOSNAP\\full_data\\test.csv')  # 替换为文件2的路径

# 使用 merge 函数筛选 SMILES 和 Target Sequence 列的值都相同的行
same_rows = pd.merge(file1, file2[['SMILES', 'Target Sequence']],
                     on=['SMILES', 'Target Sequence'],
                     how='inner')
# 筛选不同的行
different_rows = file1[~file1.set_index(['SMILES', 'Target Sequence']).index.isin(file2.set_index(['SMILES', 'Target Sequence']).index)]
# 保存相同的行到新的 CSV 文件
same_rows.to_csv('D:\\Shaun\\MolTrans-master\\dataset\\BIOSNAP_Add\\repeat(DAVIS).csv', index=False)
# 保存不同的行到新的 CSV 文件
different_rows.to_csv('D:\\Shaun\\MolTrans-master\\dataset\\BIOSNAP_Add\\norepeat(DAVIS).csv', index=False)

# 保存任一项相同的，其它的也保存

file1 = pd.read_csv('D:\\Shaun\\MolTrans-master\\dataset\\BIOSNAP_Add\\norepeat(DAVIS).csv')  # 替换为文件1的路径
file2 = pd.read_csv('D:\\Shaun\\MolTrans-master\\dataset\\BIOSNAP\\full_data\\test.csv')  # 替换为文件2的路径
# 提取文件2中的 SMILES 和 Target Sequence 列的值
file2_smiles = set(file2['SMILES'])
file2_target_sequence = set(file2['Target Sequence'])
# 筛选文件1中 SMILES 或 Target Sequence 在文件2中出现过的行
#filtered_rows = file1[file1['SMILES'].isin(file2_smiles) ]##############################################################
filtered_rows = file1[file1['SMILES'].isin(file2_smiles) | file1['Target Sequence'].isin(file2_target_sequence)]
# 筛选文件1中 SMILES 和 Target Sequence 在文件2中未出现过的行
#unfiltered_rows = file1[~file1['SMILES'].isin(file2_smiles)]
unfiltered_rows = file1[~(file1['SMILES'].isin(file2_smiles) | file1['Target Sequence'].isin(file2_target_sequence))]
# 保存出现的行到新的 CSV 文件
filtered_rows.to_csv('D:\\Shaun\\MolTrans-master\\dataset\\BIOSNAP_Add\\exit_true(DAVIS).csv', index=False)
# 保存未出现的行到新的 CSV 文件
unfiltered_rows.to_csv('D:\\Shaun\\MolTrans-master\\dataset\\BIOSNAP_Add\\exit_false(DAVIS).csv', index=False)
'''

# 每隔10行抽取一行
'''
input_file = 'D:\\Shaun\\MolTrans-master\\dataset\\DDI\\add_true.csv'
df = pd.read_csv(input_file)
# 创建两个空的DataFrame来分别存储抽取的行和未抽取的行
sampled_df = pd.DataFrame(columns=df.columns)
non_sampled_df = pd.DataFrame(columns=df.columns)


step = 10 # 每隔10行抽取一行
for i in range(len(df)):
    if i % step == 0:
        # 如果是每隔10行的那一行，则添加到sampled_df
        sampled_df = pd.concat([sampled_df, df.iloc[[i]]], ignore_index=True)
    #else:
        # 否则，添加到non_sampled_df
        #non_sampled_df = pd.concat([non_sampled_df, df.iloc[[i]]], ignore_index=True)

# 将抽取的行保存到新的CSV文件
sampled_output_file = 'D:\\Shaun\\MolTrans-master\\dataset\\DDI\\add.csv'
sampled_df.to_csv(sampled_output_file, index=False)
# 将未抽取的行保存到另一个新的CSV文件
#non_sampled_output_file = 'D:\\Shaun\\MolTrans-DTA\\dataset\\Kd1\\train.csv'
#non_sampled_df.to_csv(non_sampled_output_file, index=False)
print(f"抽取的行已保存至文件：{sampled_output_file}")
#print(f"未抽取的行已保存至文件：{non_sampled_output_file}")
'''

# 抽取拼接（也可直接做拼接: num_rows_to_sample = len(df)）
'''
# 读取CSV文件
file_path = 'D:\\Shaun\\MolTrans-DTA\\dataset\\Ki\\test.csv' #Add
df = pd.read_csv(file_path)
# 指定要随机抽取的行数
num_rows_to_sample = len(df) # 这里以10行为例
# 如果数据集小于要抽取的行数，则进行警告或处理
if num_rows_to_sample > len(df):
    print("警告：数据集小于要抽取的行数，将返回整个数据集。")
    sampled_df = df
else:
    # 随机抽取指定行数的数据（不放回）
    sampled_indices = np.random.choice(df.index, size=num_rows_to_sample, replace=False)
    sampled_df = df.loc[sampled_indices]
# 重置索引（如果需要）
sampled_df.reset_index(drop=True, inplace=True)
# 保存抽取的行到新的CSV文件
#sampled_df.to_csv('E:/Shaun/MolTrans-master/dataset/Select/train_correct.csv', index=False)
# 读取原有的train.csv文件
train_file_path = 'D:\\Shaun\\MolTrans-DTA\\dataset\\Ki\\train_origin.csv' #origin
train_df = pd.read_csv(train_file_path)
# 将抽取的数据追加到train_df的末尾
combined_df = pd.concat([train_df, sampled_df], ignore_index=True)
# 保存组合后的数据到新的CSV文件
combined_df.to_csv('D:\\Shaun\\MolTrans-DTA\\dataset\\Ki\\all.csv', index=False)
'''

# 读取CSV文件按9:1分割
'''
file_path = 'D:\\Shaun\\MolTrans-master\\dataset\\BindingDB\\train(origin).csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 按照9:1的比例划分数据
train_data = data.sample(frac=0.95, random_state=1)  # 90%的数据作为训练集
test_data = data.drop(train_data.index)             # 剩余10%的数据作为测试集

# 保存划分后的数据到新的CSV文件
#train_file_path = 'D:\\Shaun\\MolTrans-master\\dataset\\BindingDB\\train(origin).csv'  # 训练集文件路径
test_file_path = 'D:\\Shaun\\MolTrans-master\\dataset\\BindingDB\\train(Miss95%)1.csv'    # 验证集文件路径

#train_data.to_csv(train_file_path, index=False)  # 保存训练集，不包含索引
test_data.to_csv(test_file_path, index=False)    # 保存验证集，不包含索引
'''

'''
# 定义输入和输出文件路径
input_file_path = 'D:\\Shaun\\BACPI-master\\data\\affinity\\IC50\\test.txt'
output_file_path = 'D:\\Shaun\\MolTrans-DTA\\dataset\\IC50\\test.csv'

# 打开输入文件读取内容，并打开输出文件准备写入
with open(input_file_path, 'r', encoding='utf-8') as infile, \
        open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
    # 创建一个csv写入器，并写入列名
    writer = csv.writer(outfile)
    writer.writerow(['Index', 'SMILES', 'Target Sequence', 'Label'])

    # 读取输入文件的每一行，并处理成csv格式写入输出文件
    index = 0  # 初始化序号
    for line in infile:
        # 去除每行末尾的换行符，并按逗号分割成列表
        parts = line.strip().split(',')

        # 确保有三部分（两个序列和一个指标）
        if len(parts) == 3:
            smiles, target_sequence, label = parts

            # 将序号、SMILES、Target Sequence和Label写入csv文件
            writer.writerow([index, smiles, target_sequence, label])

            # 序号加1
            index += 1
        else:
            print(f"Warning: Line {index + 1} does not have exactly 3 parts and will be skipped.")
'''

'''
# 读取CSV文件
df = pd.read_csv('D:/Shaun/MolTrans-master/ESPF_A/subword_units_map_uniprot1.csv')
# 根据“index”列中字符串的长度过滤行，只保留长度为1, 3, 4, 5的行
filtered_df = df[df['index'].str.len().isin([1, 2, 3, 4, 5])]

# 处理长度大于5的字符串，拆分为长度为5的子序列
def split_into_chunks(seq, chunk_size):
    return [seq[i:i + chunk_size] for i in range(0, len(seq) - chunk_size + 1, chunk_size)]

# 创建一个新的DataFrame来存储拆分后的数据
extra_rows = []

# 遍历原始数据，处理长度大于3的index
for _, row in df[df['index'].str.len() > 3].iterrows():
    chunks = split_into_chunks(row['index'], 3)
    frequency = row['frequency']  # 假设CSV中有一个名为'frequency'的列
    for chunk in chunks:
        # 只添加长度为3的子序列
        if len(chunk) == 3:
            extra_rows.append({'index': chunk, 'frequency': frequency})
# 遍历原始数据，处理长度大于5的index
for _, row in df[df['index'].str.len() > 4].iterrows():
    chunks = split_into_chunks(row['index'], 4)
    frequency = row['frequency']  # 假设CSV中有一个名为'frequency'的列
    for chunk in chunks:
        # 只添加长度为4的子序列
        if len(chunk) == 4:
            extra_rows.append({'index': chunk, 'frequency': frequency})
# 遍历原始数据，处理长度大于5的index
for _, row in df[df['index'].str.len() > 5].iterrows():
    chunks = split_into_chunks(row['index'], 5)
    frequency = row['frequency']  # 假设CSV中有一个名为'frequency'的列
    for chunk in chunks:
        # 只添加长度为5的子序列
        if len(chunk) == 5:
            extra_rows.append({'index': chunk, 'frequency': frequency})

# 将拆分后的数据转换为DataFrame
extra_df = pd.DataFrame(extra_rows)
# 将原始过滤后的数据和拆分后的数据合并
final_df = pd.concat([filtered_df, extra_df], ignore_index=True)

# 标记重复的index行（不是第一个出现的）
final_df['duplicated'] = final_df.duplicated(subset='index', keep='first')
# 计算每个index对应的frequency总和，并将这个总和广播回原始DataFrame的相应行
final_df['total_frequency'] = final_df.groupby('index')['frequency'].transform('sum')
# 创建一个新的DataFrame来保存处理后的结果
# 只保留不是重复的行，或者保留所有行但是frequency使用总和
result_df = final_df[~final_df['duplicated']].drop(columns=['duplicated', 'frequency'])
result_df = result_df.rename(columns={'total_frequency': 'frequency'})

# 将处理后的结果保存到新的CSV文件中
result_df.to_csv('D:/Shaun/MolTrans-master/ESPF_A/subword_units_map_uniprot2.csv', index=False)
'''


# 筛选出SMILES列中不包含'.'的行
'''
# 读取CSV文件
df = pd.read_csv('D:\\Shaun\\MolTrans-DTA\\dataset\\Kd1\\test_o.csv')
# 筛选出SMILES列中不包含'.'的行
filtered_df = df[~df['SMILES'].str.contains("\\.")]
# 将筛选后的数据保存到新的CSV文件
filtered_df.to_csv('D:\\Shaun\\MolTrans-DTA\\dataset\\Kd1\\test.csv', index=False)
'''

# 删除file1中本身存在的重复行，只保留一个
'''file1 = 'D:/Shaun/MolTrans-master/dataset/DAVIS_Select/train_origin.csv'
df1 = pd.read_csv(file1)
# 保存过滤后的DataFrame到一个新的CSV文件
#unique_df1 = df1.drop_duplicates()
# 将df1重复两倍
doubled_df1 = pd.concat([df1, df1], ignore_index=True)
doubled_df1.to_csv('D:/Shaun/MolTrans-master/dataset/DAVIS_Select/train_double.csv', index=False)
print("已删除file1中本身存在的重复行，并保存为新的CSV文件。")'''
# 读取两个CSV文件,删除file1中与file2重复的行
'''file1 = 'D:/Shaun/MolTrans-master/dataset/DAVIS/train.csv'
file2 = 'D:/Shaun/MolTrans-master/dataset/DAVIS/test.csv'
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
# 使用merge方法找到重复的行，并标记合并的来源
merged_df = pd.merge(df1, df2, on=['SMILES', 'Target Sequence'], how='outer', indicator=True)
# 过滤出只在file1中出现的行（即_merge列值为'left_only'的行）
unique_df1 = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
# 保存过滤后的DataFrame到一个新的CSV文件
unique_df1.to_csv('D:/Shaun/MolTrans-master/dataset/DAVIS/train_no.csv', index=False)
print("已删除与file2重复的行，并保存为新的CSV文件。")'''
'''# 读取两个CSV文件
file1 = 'D:/Shaun/MolTrans-master/dataset/DAVIS_Select/train.csv'
file2 = 'D:/Shaun/MolTrans-master/dataset/DAVIS_Select/test.csv'
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
# 检查两个DataFrame中是否有完全相同的行（基于SMILES和Target Sequence列）
# 使用merge方法，指定内连接（'inner'），并检查合并后的DataFrame是否为空
merged_df = pd.merge(df1, df2, on=['SMILES', 'Target Sequence'], how='inner')
# 如果merged_df不为空，则说明存在相同的行
if not merged_df.empty:
    print("存在相同的行:")
    print(merged_df['SMILES'])
    print(merged_df['Target Sequence'])
else:
    print("不存在相同的行。")'''

'''
file1 = 'D:/Shaun/MolTrans-DTA/dataset/Ki/train.csv'
file2 = 'D:/Shaun/MolTrans-DTA/dataset/Kd/test.csv'
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
# 为了方便起见，我们可以重命名'Unnamed: 0'列
df1.rename(columns={'Index': 'Index1'}, inplace=True)
df2.rename(columns={'Index': 'Index2'}, inplace=True)
# 使用merge方法找到SMILES和Target Sequence同时一致的行
merged_df = pd.merge(df1, df2, on=['SMILES', 'Target Sequence'], how='inner')
# 检查是否有匹配的行
if not merged_df.empty:
    print("存在匹配的行:")
    # 打印匹配行的SMILES, Target Sequence以及在两个文件中的索引
    print(merged_df[['Index1', 'Index2']])
else:
    print("不存在匹配的行。")'''


'''# 读取第一个CSV文件，包含header
df1 = pd.read_csv('E:/Shaun/MolTrans-master/dataset/DAVIS_BindingDB/train.csv', index_col=0)

# 读取第二个和第三个CSV文件，不包含header
df2 = pd.read_csv('E:/Shaun/MolTrans-master/dataset/DAVIS_BindingDB/val.csv', header=None, index_col=0)
df3 = pd.read_csv('E:/Shaun/MolTrans-master/dataset/DAVIS_BindingDB/test.csv', header=None, index_col=0)

# 由于df2和df3没有列名，我们可以将df1的列名赋给它们
df2.columns = df1.columns
df3.columns = df1.columns

# 合并DataFrame
df_merged = pd.concat([df1, df2, df3], ignore_index=False)

# 保存合并后的DataFrame为一个新的CSV文件
df_merged.to_csv('E:/Shaun/MolTrans-master/dataset/DAVIS_BindingDB/all.csv')'''



'''# 读取两个CSV文件
df1 = pd.read_csv('E:/Shaun/MolTrans-master/dataset/DAVIS/train.csv')
df2 = pd.read_csv('E:/Shaun/MolTrans-master/dataset/BindingDB/train.csv')

# 随机抽取每个文件的10%
#sample1 = df1.sample(frac=0.1)
#sample2 = df2.sample(frac=0.1)

# 计算总行数
total_rows = len(df1) + len(df2)
# 计算每个文件需要抽取的行数（5%）
sample_size1 = int(total_rows * 0.1 * 0.3501474287427743) # drug:0.5661228064484738 target:0.3501474287427743
sample_size2 = int(total_rows * 0.1 * (1-0.3501474287427743))
# 从每个文件中随机抽取指定数量的行
sample1 = df1.sample(n=sample_size1)
sample2 = df2.sample(n=sample_size2)

# 读取第三个CSV文件
df3 = pd.read_csv('E:/Shaun/MolTrans-master/dataset/Human/train.csv')
# 将抽取的样本添加到第三个CSV文件中
combined_df = pd.concat([df3, sample1, sample2], ignore_index=True)
# 保存为一个新的CSV文件
combined_df.to_csv('E:/Shaun/MolTrans-master/dataset/Test/train_target.csv', index=False)'''



# 抽取部分数据并保存成新的csv文件
'''df = pd.read_csv('E:/Shaun/MolTrans-master/dataset/BindingDB/train.csv')
sample_size = round(len(df) * 0.1)
sample_df = df.sample(n=sample_size)
sample_df.to_csv('E:/Shaun/MolTrans-master/dataset/Test/data2_1.csv', index=False)'''


'''data = pd.read_csv('E:/Shaun/MolTrans-master/dataset/Davis/train.csv')
# 分割数据
data_1 = data[data['Label'] == 1]
data_0 = data[data['Label'] == 0]
# 选择数据
selected_1 = data_1.head(int(len(data_1) * 0.1))
selected_0 = data_0.head(int(len(data_0) * 0.1))
# 组合新文件
new_data = pd.concat([selected_1, selected_0])
new_data.to_csv('E:/Shaun/MolTrans-master/dataset/Test/head.csv', index=False)
df_b = pd.read_csv('E:/Shaun/MolTrans-master/dataset/BindingDB/train.csv')
# 将抽取的数据添加到df_b中
df_combined = pd.concat([df_b, new_data], ignore_index=True)
# 保存合并后的数据到一个新的CSV文件
df_combined.to_csv('E:/Shaun/MolTrans-master/dataset/Test/train_head.csv', index=False)'''

# 抽取部分旧数据并引入
'''df_a = pd.read_csv('E:/Shaun/MolTrans-master/dataset/Davis/train.csv')
df_b = pd.read_csv('E:/Shaun/MolTrans-master/dataset/BindingDB/train.csv')
# 从df_a中随机抽取10%的数据
df_sample = df_a.sample(frac=0.1)
# 将抽取的数据添加到df_b中
df_combined = pd.concat([df_b, df_sample], ignore_index=True)
# 保存合并后的数据到一个新的CSV文件
df_combined.to_csv('E:/Shaun/MolTrans-master/dataset/Test/train_random.csv', index=False)'''

# txt转csv
'''# 输入和输出文件名
input_filename = 'E:/Shaun/MCL-DTI-main/data/Human/Human_val.txt'
output_filename = 'E:/Shaun/MolTrans-master/dataset/Human/val.csv'
# 打开输入文件和输出文件
with open(input_filename, 'r') as infile, open(output_filename, 'w', newline='') as outfile:
    # 创建CSV写入器，‌指定列名
    writer = csv.writer(outfile)
    writer.writerow(['index', 'SMILES', 'Target Sequence', 'Label'])
    # 读取每一行，‌分割并写入CSV
    index = 0
    for line in infile:
        smiles, target_seq, label = line.strip().split()
        writer.writerow([index, smiles, target_seq, label])
        index += 1
print(f'转换完成，‌已生成{output_filename}')'''



'''old_symbol = '[Dy]'
new_symbol = ''
df = pd.read_csv('D:/MolTrans-master/dataset/BELKA/train_simple.csv')
print('CSV文件内容已读取。')
df['molecule_smiles'] = df['molecule_smiles'].replace(old_symbol, new_symbol)
print('CSV文件内容已替换。')
df.to_csv('F:/train_simple_withoutDy1.csv', index=False)
print('已生成新的CSV文件。')
df = pd.read_csv('F:/train_simple_withoutDy1.csv')
print('CSV文件内容已读取。')
# csv转parquet
# 将DataFrame转换为pyarrow的Table对象
table = pa.Table.from_pandas(df)
# 将Table保存为Parquet文件
pq.write_table(table, 'D:/MolTrans-master/dataset/BELKA/train_simple_withoutDy1.parquet')
print('CSV文件内容已写入parquet文件。')'''


# 数据抽取
'''# 设置随机种子以获得可重复的结果（可选）
random.seed(42)
# 初始化抽取的行列表
sampled_rows_list = []
# 使用chunksize分块读取大文件
chunksize = 100  # 可以根据需要调整块大小:295246830/1000=295247
sample_size = 1
i = 1
for chunk in pd.read_csv('D:/MolTrans-master/dataset/BELKA/train_simple.csv', chunksize=chunksize):
    # 在每个块中随机抽取sample_size行
    sampled_chunk = chunk.sample(n=sample_size, replace=False)
    sampled_rows_list.append(sampled_chunk)
    print(f'\r{i}/2952469', end='')  # \r将光标移回行首，end=''防止自动换行
    i = i + 1
# 合并所有块中的抽取行
sampled_rows = pd.concat(sampled_rows_list, ignore_index=True)
# 如果需要将抽取的行保存到一个新的CSV文件
sampled_rows.to_csv('D:/MolTrans-master/dataset/BELKA/train_binds0_100.csv', index=False)
print("Done!")'''


# 数据映射和保留训练使用列
'''# 读取CSV文件
df = pd.read_csv('D:/MolTrans-master/dataset/BELKA/train_binds_is1.csv')
#df = sampled_rows #df = pd.read_csv('train_1000to1.csv')
# 定义映射规则
mapping1 = {'BRD4': 'ETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEETEIMIVQAKGRGRGRKETGTAKPGVSTVPNTTQASTPPQTQTPQPNPPPVQATPHPFPAVTPDLIVQTPVMTVVPPQPLQTPPPVPPQPQPPPAPAPQPVQSHPPIIAATPQPVKTKKGVKRKADTTTPTTIDPIHEPPSLPPEPKTTKLGQRRESSRPVKPPKKDVPDSQQHPAPEKSSKVSEQLKCCSGILKEMFAKKHAAYAWPFYKPVDVEALGLHDYCDIIKHPMDMSTIKSKLEAREYRDAQEFGADVRLMFSNCYKYNPPDHEVVAMARKLQDVFEMRFAKMPDE', 'sEH': 'EDLHDKSELTDLALANAYGQYNHPFIKENIKSDEISGEKDLIFRNQGDSGNDLRVKFATADLAQKFKNKNVDIYGASFYYKCEKISENISECLYGGTTLNSEKLAQERVIGANVWVDGIQKETELIRTNKKNVTLQELDIKIRKILSDKYKIYYKDSEISKGLIEFDMKTPRDYSFDIYDLKGENDYEIDKIYEDNKTLKSDDISHIDVNLYTKKKV', 'HSA': 'DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL'}
# 使用map()函数根据映射规则创建新列
df['Target Sequence'] = df['protein_name'].map(mapping1)
# 创建一个新的 DataFrame，只包含特定的列
selected_columns = ['molecule_smiles', 'Target Sequence', 'binds']
df_simple = df[selected_columns]
# 将新的数据集保存到CSV文件中
df_simple.to_csv('D:/MolTrans-master/dataset/BELKA/train_binds_is1_simple.csv', index=False)
print("Done!")'''


# 小数据集划分
'''#file_path = 'E:/BELKA/test.csv'
# 读取CSV文件的前1000行
# df1 = pd.read_csv(file_path, nrows=5900)
# 从第2行开始读取，到第4行结束
#df2 = pd.read_csv(file_path, skiprows=1048575, nrows=1674896-1048575)
# 计算每个部分的行数
df = df_simple
total_rows = len(df)
print(total_rows)
train_rows = int(0.8 * total_rows)
test_rows = total_rows - train_rows
# 打乱数据行
df = df.sample(frac=1)
# 拆分数据
train_df = df.head(train_rows)
test_df = df.tail(test_rows)
# 将拆分后的数据保存到新的CSV文件
train_df.to_csv('D:/MolTrans-master/dataset/BELKA/train5000to1.csv', index=False)
test_df.to_csv('D:/MolTrans-master/dataset/BELKA/test5000to1.csv', index=False)
'''


# 替换符号和列名
'''
old_symbol = '[Dy]'
new_symbol = ''
target_column = 'molecule_smiles'
# 读取原始CSV文件
with open('D:/MolTrans-master/dataset/BELKA/train_simple.csv', mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    rows = [row for row in reader]
# 处理指定列中的符号
for row in rows:
    if target_column in row and old_symbol in row[target_column]:
        row[target_column] = row[target_column].replace(old_symbol, new_symbol)
# 写入新的CSV文件
with open('E:/train_simple_withoutDy1.csv', mode='w', newline='', encoding='utf-8') as outfile:
    fieldnames = reader.fieldnames  # 保留原始文件的列名
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()  # 写入表头
    writer.writerows(rows)  # 写入处理后的数据行
print("Done!")

# csv转parquet
df = pd.read_csv('E:/train_simple_withoutDy1.csv')
# 将DataFrame转换为pyarrow的Table对象
table = pa.Table.from_pandas(df)
# 将Table保存为Parquet文件
pq.write_table(table, 'D:/MolTrans-master/dataset/BELKA/train_simple_withoutDy1.parquet')
print('CSV文件内容已写入parquet文件。')'''

# csv转txt
'''csv_file_path = 'D:/MolTrans-master/dataset/BELKA/train1000to1_withoutDy.csv'
txt_file_path = 'D:/MolTrans-master/dataset/BELKA/train1000to1_withoutDy.txt'
print("Load Done!")
# 打开CSV文件以读取内容
with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
    # 创建一个CSV读取器
    reader = csv.reader(csvfile)
    # 跳过第一行（列名）
    next(reader)
    # 打开TXT文件以写入内容
    with open(txt_file_path, 'w', encoding='utf-8') as txtfile:
        # 遍历CSV文件的每一行
        for row in reader:
            # 将每行的内容（即每一列）中的[Dy]替换为空或空格
            modified_row = [col.replace('[Dy]', '') for col in row]
            # 将替换后的每行内容（即每一列）用逗号连接，并写入TXT文件
            txtfile.write(','.join(modified_row) + '\n')
print('CSV文件内容已写入TXT文件，并跳过了第一行且替换了[Dy]。')'''
