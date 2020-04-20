# 第二章：Tensorflow2.0基础与进阶
## Eager动态图
### Eager模式
Eager Execution，动态图，使得用户可以在不创建Graph（图）的情况下运行Tensorflow代码。
之前在Ubuntu上安装过1.X的版本，就是在打印的时候是不会直接显示出数值的，现在的使用方法会简化一些。我的直观想法就是这样。
接下来跟着教材实践一下。
下面默认启用了Eager模式，使用tensorflow读入一个序列然后打印。
```python
import tensorflow as tf
data=tf.constant([1,2])
print(data)
```
这时候就会打印出下面内容，表示这个Tensor数据格式，数值[1,2]，维度是2，数据类型int32。
```
tf.Tensor([1 2], shape=(2,), dtype=int32)
```
如果使用数据的numpy()函数，就会转换为常数格式。
```python
import tensorflow as tf
data=tf.constant([1,2])
print(data)
print(data.numpy())
```
打印如下。
```
tf.Tensor([1 2], shape=(2,), dtype=int32)
[1 2]
```
### 数据的读取
使用Dataset API完成数据迭代。
```python
import tensorflow as tf
import numpy as np
arr_list=np.arange(0,50).astype(np.float32)
#np.arrange()  起点，终点，步长  生成的时候默认int32，为了计算准确，转化为float32
shape=arr_list.shape
#使用dataset读取API
dataset=tf.data.Dataset.from_tensor_slices(arr_list) #读取数据
dataset_iterator=dataset.shuffle(shape[0]).batch(10)
#使用shuffle打乱顺序，然后每个batch（批）为10输出
def model(xs):
    outputs=tf.multiply(xs,0.5)
    return outputs
for it in dataset_iterator:
    logits=model(it)
    print(logits)
```
然后输出以下，打乱顺序的以10个为一组，再乘上0.5的结果。
```
tf.Tensor([10.5 11.5 14.5  7.5 16.   6.5 10.  21.5 20.5 23. ], shape=(10,), dtype=float32)
tf.Tensor([15.5 17.5 14.  11.  24.5 21.   4.  16.5  3.5  1. ], shape=(10,), dtype=float32)
tf.Tensor([12.  19.5 23.5 17.   8.   4.5  8.5 12.5  1.5  2.5], shape=(10,), dtype=float32)
tf.Tensor([ 5.   3.  18.  19.  20.   6.  13.5  7.   0.5  0. ], shape=(10,), dtype=float32)
tf.Tensor([22.  22.5 18.5  9.   9.5 24.   5.5  2.  13.  15. ], shape=(10,), dtype=float32)
```
### 线性回归例子
总结一下就是以下几个步骤。
1. 这个模型是什么，这里就是一元线性回归。
2. 损失函数loss。这里考虑的就是均方差（MSE）表示拟合数据和真实数据的误差。
3. 梯度函数的更新。直接调用优化器，这个梯度函数我还不是很了解。
