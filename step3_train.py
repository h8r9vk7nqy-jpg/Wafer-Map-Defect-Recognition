import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 50)
print(">>> [Step 3] 启动 CNN 模型训练程序")
print(f">>> TensorFlow 版本: {tf.__version__}")
print("=" * 50)

# 1. 加载数据
print(">>> 正在加载 processed_data.npz ...")
# 确保这个 .npz 文件就在你旁边
try:
    data = np.load('processed_data.npz')
except FileNotFoundError:
    print("!!! 错误：找不到 processed_data.npz，请检查 Step 2 是否成功运行。")
    exit()

X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

print(f">>> 数据加载完成。训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 2. 搭建 CNN 模型
model = models.Sequential([
    # 第一层卷积
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    # 第二层卷积
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # 第三层卷积
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # 展平与全连接
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # 防止过拟合
    layers.Dense(9, activation='softmax') # 9类输出
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. 开始训练
print("-" * 30)
print(">>> 开始训练模型 (Epochs=10)...")
# Mac M系列芯片训练这个量级的数据，大概需要 2-5 分钟
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test),
                    batch_size=64)

# 4. 保存模型
model.save('wafer_model.h5')
print(">>> 模型已保存为 wafer_model.h5")

# 5. 画图
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.title('Loss')
plt.legend()
plt.savefig('training_history.png')
print(">>> 训练曲线已保存为 training_history.png")
print("=" * 50)