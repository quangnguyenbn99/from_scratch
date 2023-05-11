import numpy as np
import random

# Here’s a high-level overview of the perceptron learning algorithm:
	# Initialize the weights and threshold with random values.
	# For each input-output pair in the training data:
	# Compute the perceptron’s output using the current weights and threshold.
	# Update the weights and threshold based on the difference between the desired output and the perceptron’s output – the error.
	# Repeat steps 2 and 3 until the perceptron classifies all input-output pairs correctly, or a specified number of iterations have been completed.
# The update rule for the weights and threshold is simple:
	# If the perceptron’s output is correct, do not change the weights or threshold.
	# If the perceptron’s output is too low, increase the weights and decrease the threshold.
	# If the perceptron’s output is too high, decrease the weights and increase the threshold.

def perceptron(inputs,weights,threshold):
	weighted_sum = sum(x*w for x,w in zip(inputs,weights))
	return 1 if weighted_sum>= threshold else 0

def is_nonnegative(x):
	return perceptron([x],[1],0)

def not_function(x):
	weight = -1
	threshold = -0.5
	return perceptron([x], [weight], threshold)

def train_perceptron(data, learning_rate=0.1, max_iter=1000):

    # max_iter is the maximum number of training cycles to attempt
    # until stopping, in case training never converges.

    # Find the number of inputs to the perceptron by looking at
    # the size of the first input tuple in the training data:
    first_pair = data[0]
    num_inputs = len(first_pair[0])

    # Initialize the vector of weights and the threshold:
    weights = [random.random() for _ in range(num_inputs)]
    threshold = random.random()
   
    # Try at most max_iter cycles of training:
    for _ in range(max_iter):

        # Track how many inputs were wrong this time:
        num_errors = 0
        
        # Loop over all the training examples:
        for inputs, desired_output in data:
            output = perceptron(inputs, weights, threshold)
            error = desired_output - output
            
            if error != 0:
                num_errors += 1
                for i in range(num_inputs):
                    weights[i] += learning_rate * error * inputs[i]
                threshold -= learning_rate * error
        
        if num_errors == 0:
            break
    
    return weights, threshold

and_data = [
 ((0, 0),  0),
 ((0, 1),  0),
 ((1, 0),  0), 
 ((1, 1),  1)
]    
and_weights, and_threshold = train_perceptron(and_data)

print("Weights:", and_weights)
print("Threshold:", and_threshold)
print(perceptron((0,0),and_weights,and_threshold)) # prints 0
print(perceptron((0,1),and_weights,and_threshold)) # prints 0
print(perceptron((1,0),and_weights,and_threshold)) # prints 0
print(perceptron((1,1),and_weights,and_threshold)) # prints 1

or_data = [
    ((0, 0), 0),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 1)
]
or_weights, or_threshold = train_perceptron(or_data)

print("Weights:", or_weights)
print("Threshold:", or_threshold)
print(perceptron((0,0),or_weights,or_threshold)) # prints 0
print(perceptron((0,1),or_weights,or_threshold)) # prints 1
print(perceptron((1,0),or_weights,or_threshold)) # prints 1
print(perceptron((1,1),or_weights,or_threshold)) # prints 1

xor_data = [
    ((0, 0), 0),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 0)
]
xor_weights, xor_threshold = train_perceptron(xor_data, max_iter=10000)

print("Weights:", xor_weights)
print("Threshold:", xor_threshold)
print(perceptron((0,0),xor_weights,xor_threshold)) # prints 0
print(perceptron((0,1),xor_weights,xor_threshold)) # prints 1
print(perceptron((1,0),xor_weights,xor_threshold)) # prints 1
print(perceptron((1,1),xor_weights,xor_threshold)) # prints 1!!
