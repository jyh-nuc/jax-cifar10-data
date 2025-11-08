import jax
import jax.numpy as jnp

def generate_synthetic_data(train: bool = True):
    num_samples = 60000 if train else 10000
    key = jax.random.PRNGKey(42)
    images = jax.random.uniform(key, shape=(num_samples, 32, 32, 3))
    labels = jax.random.randint(key, shape=(num_samples,), minval=0, maxval=10)
    return (images, labels)  # 严格返回二元组

def create_data_loader(data: tuple, batch_size: int):
    images, labels = data
    dataset = list(zip(images, labels))
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size]

def evaluate(state, data_loader):
    correct = 0
    total = 0
    for batch in data_loader:
        x, y = zip(*batch)
        x = jnp.array(x)
        y = jnp.array(y)
        logits = state.apply_fn({'params': state.params}, x)
        preds = jnp.argmax(logits, axis=1)
        correct += jnp.sum(preds == y)
        total += len(y)
    return correct / total
