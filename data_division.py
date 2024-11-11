import os
import shutil
from sklearn.model_selection import train_test_split

data_folder = 'Data/kvasir/masks'
train_folder = 'Data/kvasir/train/masks'
test_folder = 'Data/kvasir/test/masks'

# 获取所有数据文件
data_files = os.listdir(data_folder)

# 使用 train_test_split 分割数据，test_size=0.1 表示10%的数据作为测试集
train_files, test_files = train_test_split(data_files, test_size=0.2, random_state=42)
print(len(train_files), len(test_files))
# 创建训练和测试文件夹（如果它们不存在）
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 分别复制文件到训练和测试文件夹
for file in train_files:
    shutil.copy(os.path.join(data_folder, file), os.path.join(train_folder, file))
for file in test_files:
    shutil.copy(os.path.join(data_folder, file), os.path.join(test_folder, file))