import jax
import jax.numpy as jnp
from jax import random

def generate_synthetic_data():
    key = random.PRNGKey(0)
    train_key, val_key = random.split(key)
    
    train_x = random.normal(train_key, (60000, 32, 32, 3))
    train_y = random.randint(train_key, (60000,), 0, 10)
    
    val_x = random.normal(val_key, (10000, 32, 32, 3))
    val_y = random.randint(val_key, (10000,), 0, 10)
    
    return (train_x, train_y), (val_x, val_y)

def create_data_loader(data, batch_size=32):
    x, y = data
    num_batches = len(x) // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        yield x[start:end], y[start:end]
