import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import time

# 设置随机种子
np.random.seed(42)

print("=" * 50)
print(">>> [Step 2] 启动数据预处理程序 (修复版)")
print("=" * 50)

# 1. 读取原始数据
print(f"[{time.strftime('%H:%M:%S')}] 正在读取 LSWMD.pkl...")
try:
    df = pd.read_pickle('LSWMD.pkl')
    print(f"[{time.strftime('%H:%M:%S')}] 读取成功！原始数据量: {len(df)}")
except FileNotFoundError:
    print("!!! 错误: 找不到 LSWMD.pkl 文件。")
    exit()

# 2. 数据清洗 (Filtering) - 修复版
print("-" * 30)
print(">>> 正在剔除无标签及损坏的数据...")

# 初步剔除 NaN
df = df[df['failureType'].notnull()]
df = df.reset_index(drop=True)

# --- 核心修复开始 ---
# 定义一个安全提取函数，专门对付 [] 这种空列表
def safe_get_label(x):
    try:
        # 尝试正常提取 [['Loc']] -> 'Loc'
        if len(x) > 0 and len(x[0]) > 0:
            return x[0][0]
        else:
            return None # 如果是 []，返回 None
    except:
        return None

# 应用安全函数
df['failureType'] = df['failureType'].apply(safe_get_label)

# 再次剔除那些提取出来是 None 的行 (即原本是 [] 的行)
df = df[df['failureType'].notnull()]
df = df.reset_index(drop=True)
# --- 核心修复结束 ---

print(f">>> 清洗完毕。剩余有效有标签样本数: {len(df)}")
print(f">>> 类别分布概览:\n{df['failureType'].value_counts()}")

# 3. 图像统一尺寸 (Resizing)
TARGET_SIZE = 64
print("-" * 30)
print(f">>> 正在将所有晶圆图统一缩放至 {TARGET_SIZE}x{TARGET_SIZE}...")
print(">>> 这一步涉及大量图像运算，可能会花 1-2 分钟，请耐心等待...")

def resize_wafer(wafer_map):
    img = Image.fromarray(wafer_map)
    img = img.resize((TARGET_SIZE, TARGET_SIZE), resample=Image.NEAREST)
    return np.array(img)

# 计时
start_time = time.time()
df['waferMap64'] = df['waferMap'].apply(resize_wafer)
end_time = time.time()
print(f">>> 缩放完成！耗时: {end_time - start_time:.2f} 秒")

# 4. 构建数据集 (X 和 y)
print("-" * 30)
print(">>> 正在构建训练矩阵 (Normalization & One-hot)...")

X = np.stack(df['waferMap64'].values)
X = X.reshape(-1, TARGET_SIZE, TARGET_SIZE, 1)
X = X / 2.0 # 归一化

y = df['failureType'].values

# 标签映射字典
mapping = {
    'Center': 0, 
    'Donut': 1, 
    'Edge-Loc': 2, 
    'Edge-Ring': 3, 
    'Loc': 4, 
    'Near-full': 5, 
    'Random': 6, 
    'Scratch': 7, 
    'none': 8
}
# 转换标签
y_int = np.array([mapping[label] for label in y])

# 5. 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y_int, test_size=0.2, random_state=42)

print("-" * 30)
print(f"训练集 (X_train) 形状: {X_train.shape}")
print(f"测试集 (X_test)  形状: {X_test.shape}")

# 6. 保存成果
print("-" * 30)
print(">>> 正在保存处理好的数据到 processed_data.npz ...")
np.savez('processed_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print("=" * 50)
print(">>> [Success] Step 2 (修复版) 全部完成！")
print("=" * 50)