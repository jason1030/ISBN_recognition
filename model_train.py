from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl
import sys
import csv
from sklearn.model_selection import train_test_split

# solve could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# 不同库版本，使用此代码块查看
print(sys.version_info)
for module in mpl, np, tf, keras:
    print(module.__name__, module.__version__)

# load train and test
with open('features.csv', encoding='utf-8') as f:
    X = np.loadtxt(f, delimiter=",")
with open('labels.csv', encoding='utf-8') as g:
    Y = np.loadtxt(g, str, delimiter=",")
rows = X.shape[0]
X = X.reshape([rows, 28, 28])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=30)
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = X_train.astype("float32") / 255
x_test = X_test.astype("float32") / 255

# 1 Byte = 8 Bits，2^8 -1 = 255。[0,255]代表图上的像素，同时除以一个常数进行归一化。1 就代表全部涂黑。0 就代表没涂

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# CNN 的输入方式必须得带上channel，这里扩充一下维度

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 17)
y_test = keras.utils.to_categorical(y_test, 17)

# y 属于 [0,9]代表手写数字的标签，这里将它转换为0-1表示，可以类比one-hot，举个例子，如果是2

# [[0,0,1,0,0,0,0,0,0,0]……]

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        # keras.layers.Dropout(0.2),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        # keras.layers.Dropout(0.2),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=17, activation="softmax"),
    ]
)

# 注意，Conv2D里面有激活函数不代表在卷积和池化的时候进行。而是在DNN里进行，最后拉直后直接接softmax就行


# kernel_size 代表滤波器的大小，pool_size 代表池化的滤波器的大小

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

history = model.fit(x_train, y_train, batch_size=5, epochs=15, validation_split=0.1)  # 10层交叉检验
model.save('my_ISBN_model')
# re_model = keras.models.load_model('my_ISBN_model')
# score = model.evaluate(x_test, y_test)
# # re_score = re_model.evaluate(x_test, y_test)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])
# print("re_test loss", re_score[0])
# print("re_test accuracy", re_score[1])


# Test loss: 0.03664601594209671
# Test accuracy: 0.989300012588501

# visualize accuracy and loss
def plot_(history, label):
    plt.plot(history.history[label])
    plt.plot(history.history["val_" + label])
    plt.title("model " + label)
    plt.ylabel(label)
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


# plot_(history, "accuracy")
# plot_(history, "loss")
