### A
According to Moroney, machine learning and traditional programming consist of three similar components: data, answers, and rules. The difference between these two types of programming is the order in which the components are used. In machine learning, data and answers are used as inputs to generate rules. In traditional programming, data and rules are used as inputs to generate answers.

### B
```python
import numpy as np
import tensorflow as tf
from tf import keras

model = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

xs = np.array([1,2,3,4,5])
ys = np.array([2,5,8,11,14])
model.fit(xs, ys, epochs = 500)

print(model.predict([])
```

### C
```python
import numpy as np
import tensorflow as tf
from tf import keras

model = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

xs = np.array([4.0, 3.0, 4.0, 5.0, 2.0, 3.0], dtype = float)
ys = np.array([2.89, 2.29, 3.99, 3.475, 2.5, 0.97], dtype = float)
model.fit(xs, ys, epochs = 1000)

print(model.predict([4.0])
```

