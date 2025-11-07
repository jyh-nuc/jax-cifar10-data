import tensorflow_datasets as tfds
import tensorflow as tf

def load_cifar10(batch_size=64):
    # 禁用GPU
    tf.config.set_visible_devices([], 'GPU')
    
    # 加载CIFAR-10数据集
    train_ds, test_ds = tfds.load(
        'cifar10', 
        split=['train', 'test'], 
        as_supervised=True,
        shuffle_files=True
    )
    
    # 预处理函数
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - 0.5) * 2.0
        return image, label
    
    # 应用预处理并批处理
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    
    return train_ds, test_ds