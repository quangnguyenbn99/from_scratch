import math

import torch
import torch.nn as nn
from typing import List

def create_binary_list_from_int(number: int) -> List[int]:
    if number < 0 or type(number) is not int:
        raise ValueError("Only Positive integers are allowed")

    return [int(x) for x in list(bin(number))[2:]]


def generate_even_data(max_int: int, batch_size: int=16) -> Tuple[List[int], List[List[int]]]:
    # Get the number of binary places needed to represent the maximum number
    max_length = int(math.log(max_int, 2))

    # Sample batch_size number of integers in range 0-max_int
    sampled_integers = np.random.randint(0, int(max_int / 2), batch_size)

    # create a list of labels all ones because all numbers are even
    labels = [1] * batch_size

    # Generate a list of binary numbers for training.
    data = [create_binary_list_from_int(int(x * 2)) for x in sampled_integers]
    data = [([0] * (max_length - len(x))) + x for x in data]

    return labels, data

class Generator(nn.Module):

    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), int(input_length))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))

class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(int(input_length), 1);
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))

def train(max_int: int = 128, batch_size: int = 16, training_steps: int = 500):
    input_length = int(math.log(max_int, 2))

    # Models
    generator = Generator(input_length)
    discriminator = Discriminator(input_length)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

    # loss
    loss = nn.BCELoss()

    for i in range(training_steps):
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

# Create noisy input for generator (Need float type instead of int)
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        # Generating new “fake” data by passing the noise to the generator
        generated_data = generator(noise)

        # Generate examples of even real data
        true_labels, true_data = generate_even_data(max_int, batch_size=batch_size)
        true_labels = torch.tensor(true_labels).float()
        true_data = torch.tensor(true_data).float()

# Train the generator (Want to make it close to TRUE)
        # We invert the labels here and don't train the discriminator because we want the generator
        # to make things the discriminator classifies as true. (Get the predictions from the discriminator on the “fake” data)
        generator_discriminator_out = discriminator(generated_data)
        
        # Calculate the loss from the discriminator’s output using labels as if the data were “real” instead of fake. 
        generator_loss = loss(generator_discriminator_out, true_labels)
        generator_loss.backward()
        
        # Backpropagate the error through just the generator. 
        generator_optimizer.step()

# Train the discriminator on the true/generated data (Want to make it wise to TRUE and FAKE)
        discriminator_optimizer.zero_grad()
        
        # Pass in a batch of only data from the true data set with a vector of all one labels.
        true_discriminator_out = discriminator(true_data)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)

        # adding .detach() here to not training the generator (Do not touch the gradient from the generator).
        # Pass generated data into the discriminator, with detached weights, and zero labels
        generator_discriminator_out = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size))
        
        # Average the loss from steps one and two
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()

        # Backpropagate the gradients through just the discriminator.
        discriminator_optimizer.step()