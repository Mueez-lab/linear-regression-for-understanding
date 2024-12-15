import numpy as np

#z_score_Normalization is a part of feature scaling
#it will give us range of x_1

def z_score_Normalization(x_axis):
    mu = np.mean(x_axis) # mu is the average of x_axis
    sigma = np.std(x_axis) #sigma is standard deviation
    x_norm = (x_axis-mu)/sigma #formula to calculate z_score or range in blow +ve x and -ve x
    return x_norm, mu, sigma


#cost function is actually the difference between actual value and predicted value
def compute_cost(x_axis, y_axis,w,b):
    m = len(y_axis)
    predicted_value = np.dot(x_axis,w)+b
    j= (1/2*m) * np.sum(np.square(predicted_value - y_axis)) # so here we are multiplying m with 2 to reduce the value and taking square with np to remove -ve sign if occur                  
    return j

def gradient_descent(x_axis, y_axis, w, b, alpha, iterations):
    m = len(y_axis)  # Number of training examples
    
    print(f"{'Iteration':>10} | {'Cost':>12} | {'Weights':>40} | {'Bias':>8}") # use to create table with a width of 10,12,40,8 characters
    print("-" * 70) # 70 times to create a horizontal line
    
    for i in range(1, iterations + 1):
        predicted_value = np.dot(x_axis, w) + b
        dj_dw = (1 / m) * np.dot(x_axis.T, (predicted_value - y_axis))  # temp_w because w is used while updating value of b too and t is the transpose use to align with dit product
        dj_db = (1 / m) * np.sum(predicted_value - y_axis)  # temp_b in derivation  1/2 is removed because it is constant if we put 1/2 the it is not going to affect the outcome of gradient descent but it will affect the no of iterations 
        # update the value of  w and b
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        # changing cost at eact iteration
        cost = compute_cost(x_axis, y_axis, w, b)
        # print updates for every iteration
        print(f"{i:>10} | {cost:12.6f} | {str(w.flatten().round(4)):>40} | {b:8.4f}")    
    return w, b

   
    
x_axis = np.array([[1.0,2.0,3.0],[2.0,1.0,0.5],[0.5,1.5,2.0]])
y_axis = np.array([[14],[8],[10]])
w = np.array([[0.5],[0.3],[-0.2]]) #weight and weight represents the slope of the line
b = 1 # it is bias or y-intercept formula of b is (mean-of-y - (w * mean-of-x)
alpha = 0.01 # alpha is the learning late too big can lead to wrong min value which results in bad cost and too small can slow down
theta = w
iterations= 1000 # in documentation of a=0.01 then we should choose 1k but after 403 cost stops decreasing significantly so is it better to choose 403 or 1K?
x_norm ,mu, sigma = z_score_Normalization(x_axis)
print("Cost without gradient descent aka ",compute_cost(x_axis,y_axis,w,b))
# Perform gradient descent
w_final, b_final = gradient_descent(x_norm, y_axis, w, b, alpha, iterations)
print(f"Final Weights: \n{w_final}")
print(f"Final Bias: {b_final}")

# Compute final cost
final_cost = compute_cost(x_norm, y_axis, w_final, b_final)
print(f"Final Cost after Gradient Descent: {final_cost}")

