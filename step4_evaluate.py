import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 50)
print(">>> [Step 4] 启动模型评估与可视化程序")
print("=" * 50)

# 1. 加载数据
print(">>> 正在加载测试数据 processed_data.npz ...")
data = np.load('processed_data.npz')
X_test, y_test = data['X_test'], data['y_test']

# 2. 加载模型
print(">>> 正在加载训练好的模型 wafer_model.h5 ...")
try:
    model = models.load_model('wafer_model.h5')
    print(">>> 模型加载成功！")
except:
    print("!!! 错误：找不到模型文件。")
    exit()

# 3. 进行预测 (考试)
print(">>> 正在对 34,590 张测试图片进行预测...")
y_pred_prob = model.predict(X_test)
# 把概率变成类别 (例如 [0.1, 0.9, ...] -> 1)
y_pred = np.argmax(y_pred_prob, axis=1)

# 4. 生成分类报告 (Precision, Recall, F1-score)
target_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'None']
report = classification_report(y_test, y_pred, target_names=target_names)

print("-" * 30)
print(">>> 分类详细报告 (Classification Report):")
print(report)
print("-" * 30)

# 将报告保存到 txt 用于文书
with open("classification_report.txt", "w") as f:
    f.write(report)

# 5. 绘制混淆矩阵 (The Money Shot)
print(">>> 正在绘制混淆矩阵...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
# 使用 seaborn 画热力图
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label (AI预测)')
plt.ylabel('True Label (真实标签)')
plt.title('Confusion Matrix - Wafer Defect Recognition')
plt.tight_layout()

plt.savefig('confusion_matrix.png', dpi=300)
print(">>> 混淆矩阵已保存为 confusion_matrix.png")
print("=" * 50)
print(">>> [Success] 全流程结束！请查看生成的图片和报告。")