import numpy as np
import matplotlib.pyplot as plt

# x_train is the input variable (size in 1000 square m)
# y_train is the target (price in 1000s of dollars
x_train = np.array([1.0,2.0])
y_train = np.array([300.0, 500.0])

# m is the number of training examples
m = len(x_train)
print(f"Number of training data is: {m}")

#Training example

i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"x^({i}),y^({i}) = ({x_i}, {y_i}) ") #First row of training dataset


#Model function
w = 200
b = 100
print (f"w: {w}")
print (f"b: {b}")
def compute_model_output (x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i]+b
    
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)

plt.plot(x_train, tmp_f_wb,c ='b', label = 'Our prediction')
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

# Make a Prediction.  Let's predict the price of a house with 1200 sqft.
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")

