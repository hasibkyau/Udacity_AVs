import pandas as pd
from sklearn.model_selection import train_test_split
from utlis import *

path = 'myData'
data = importDataInfo(path)
# data = pd.DataFrame(data)
# print(len(data))
# print(type(data))
# print(data)

data = balanceData(data, display=False)
# data = pd.DataFrame(data)
print(len(data))

imagesPath, steerings = loadData(path,data)

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

model = createModel()
model.summary()

history = model.fit(batchGen(xTrain, yTrain, 100, 1),
                                  steps_per_epoch=300,
                                  epochs=10,
                                  validation_data=batchGen(xVal, yVal, 100, 0),
                                  validation_steps=200)

model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()

