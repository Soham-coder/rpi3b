import sys
print(sys.argv[0])

#PUt the paths here.
test_data = 'Test Data for RPi'
original_model = 'v3/PreConv_Spark.h5'
tflite_model = 'v3/PreConv_Spark_TFLite.tflite'
tflite_model_quantized = 'v3/PreConv_Spark_Quantized.tflite'


#Imort the packages
import time
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(tf.keras.__version__)    
#Output must be 
#2.3.0
#2.4.0
import numpy as np
import cv2
import  os
from sklearn.preprocessing import OneHotEncoder
print('All packages are imported')


def read_image(im_path):
  img = cv2.imread(im_path,0)
  img = cv2.resize(img,(240,240))
  return img
  
  
time0 = time.time()
x_test, y_test = [], []
list_folder=os.listdir(path = test_data)
for i in list_folder:
  new_path=os.path.join(test_data,i) 
  pic_list=os.listdir(new_path)                                              
  for img in pic_list:  
    im_path = os.path.join(new_path,img)
    image = read_image(im_path)
    if i == 'flair':
      label = 0
    elif i == 't1':
      label = 1
    elif i == 't1ce':
      label = 2
    elif i == 't2':
      label = 3
    x_test.append(image) 
    y_test.append(label) 


X_test = np.array(x_test).reshape(-1,240,240,1)
X_test = X_test/255
y_test = np.array(y_test).reshape(-1,1)
encoder = OneHotEncoder()
y_test = encoder.fit_transform(y_test)
Y_test = y_test.toarray()
time1 = time.time()
print("Test images reading is done")
print("Number of Test images",X_test.shape[0])
print("Time for reading and Preprocessing", time1 - time0)
print("Average time for reading and pre-processing", (time1-time0)/X_test.shape[0])


#Read the models
# Initialize TFLite interpreter using the model.
# Load TFLite model and allocate tensors.
time2 = time.time()
#original
original = tf.keras.models.load_model(original_model)
#TFLite
interpreter = tf.lite.Interpreter(model_path=tflite_model)
interpreter.allocate_tensors()
#quantized
interpreter_q = tf.lite.Interpreter(model_path=tflite_model_quantized)
interpreter_q.allocate_tensors()
time3 = time.time()
print("\nTime for Loading all models", time3 - time2)


# Get input and output tensors.
#TFLite
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#Quantized
input_details_q = interpreter_q.get_input_details()
output_details_q = interpreter_q.get_output_details()

#Evaluation
#original
time4 = time.time()
evaluation = original.evaluate(X_test, Y_test)
time5 = time.time()
print('\Accuracy of Original : ', evaluation[1])
print("Total time to predict all", time5 -time4)
print("Average time to predict: ", (time5 -time4)/len(X_test))

time6 = time.time()
n = 0  #count for accuracy
normal = []
# Test model on input data.
for i in range(1, len(X_test)):
  test_image = np.expand_dims(X_test[i], axis=0).astype(np.float32)
  interpreter.set_tensor(input_details[0]['index'], test_image)
  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  output_data = interpreter.get_tensor(output_details[0]['index'])
  
  if np.argmax(output_data)==np.argmax(Y_test[i]):
    n+=1
  else:
    normal.append(i)

time7 = time.time()
print('\n\nAccuracy of TFLite model= ', n/len(X_test))
print("Time to predict all " , len(X_test), " images is: ", time7-time6)
print("Average time to predict a image is: ", (time7-time6)/len(X_test))
    
time8 = time.time()
q = 0
quantized = []
# Test model on input data.
for i in range(1, len(X_test)):
  test_image = np.expand_dims(X_test[i], axis=0).astype(np.float32)
  interpreter_q.set_tensor(input_details_q[0]['index'], test_image)
  interpreter_q.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  output_data_q = interpreter_q.get_tensor(output_details_q[0]['index'])


  if np.argmax(output_data_q)==np.argmax(Y_test[i]):
    q+=1
  else:
    quantized.append(i)

time9 = time.time()
print('\n\nAccuracy of quantized model= ', q/len(X_test))
print("Time to predict all ", len(X_test), " images is: ", time9-time8)
print("Average time to predict a image is: ", (time9-time8)/len(X_test))

