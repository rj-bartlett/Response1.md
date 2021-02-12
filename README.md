### A
According to Moroney, machine learning and traditional programming consist of three similar components: data, answers, and rules. The difference between these two types of programming is the order in which the components are used. In machine learning, data and answers are used as inputs to generate rules. In traditional programming, data and rules are used as inputs to generate answers.

### B
```python
import numpy as np 
import tensorflow
from tensorflow import keras

model = keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

xs = np.array([-1.0,0.0,1.0,2.0,3.0,4.0], dtype = float)
ys = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0], dtype = float)

model.fit(xs,ys,epochs = 500)

print(model.predict([7.0]))
```

First result: 12.988924
Second result: 12.987505
Explaination: 

### C
