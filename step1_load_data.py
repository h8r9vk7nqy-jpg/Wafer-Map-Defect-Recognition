import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置绘图风格，让图表在 Mac 上显示更清晰
plt.style.use('ggplot') 

print(">>> 正在读取数据，文件较大 (约2GB+)，请耐心等待 1-2 分钟...")
print(">>> 如果内存不够，这一步可能会卡顿，请不要关闭程序。")

# 1. 读取数据
try:
    # 你的文件名如果是别的，请在这里修改
    df = pd.read_pickle('LSWMD.pkl')
    print(f">>> 数据读取成功！总共有 {len(df)} 个晶圆数据。")
except FileNotFoundError:
    print("!!! 错误：找不到 LSWMD.pkl 文件。请确认文件是否在代码目录下！")
    exit()

# 2. 这里的核心逻辑是：找到一张有特定缺陷的图画出来
# 这一步是为了验证你的 matplotlib 和 numpy 是否工作正常
print("-" * 30)
print(">>> 正在寻找一张带有 'Donut' (甜甜圈模式) 缺陷的晶圆...")

found = False
# 我们只遍历有标签的数据，提升速度
df_labeled = df[df['failureType'].notnull()]

for index, row in df_labeled.iterrows():
    # 数据集里的 failureType 格式比较怪，通常是 [[type, rate]]，需要拆包
    try:
        f_type = row['failureType'][0][0]
        
        if f_type == 'Donut': # 找到甜甜圈模式
            wafer_map = row['waferMap']
            
            # 开始画图
            plt.figure(figsize=(6, 6))
            # cmap='inferno' 是一种热力图配色，适合展示半导体缺陷
            plt.imshow(wafer_map, cmap='inferno') 
            plt.title(f"Wafer Index: {index} | Defect: {f_type}", fontsize=14)
            plt.colorbar()
            plt.grid(False) # 关掉网格，看图更清楚
            plt.show()
            
            print(f">>> 成功绘图！索引号: {index}")
            print(">>> 恭喜你，项目第一阶段（数据加载）已完成。")
            found = True
            break
    except Exception as e:
        continue

if not found:
    print("暂时没找到 Donut 类型，请尝试把代码里的 'Donut' 改成 'Loc' 再试一次。")