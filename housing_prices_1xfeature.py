# TODO: add validation set and validation plots
# TODO: add predict method
# TODO: add plot for predictions and animated gifs during training
# TODO: add save and load model



import numpy as np
import matplotlib.pyplot as plt
from utils import generate_linear_points_with_noise, min_max_normalize
from layers import Linear, Reshape
from activations import Tanh
from losses import MSE
from models import Sequential


# Train dataset
house_areas_range = (100, 600)
house_areas, housing_prices = generate_linear_points_with_noise(
                                                                x_range=house_areas_range,
                                                                intercept=50,
                                                                slope=1.5,
                                                                num_points=25,
                                                                noise_std_dev=30)

# Train dataset
house_areas_range_val = (700, 1000)
house_areas_val, housing_prices_val = generate_linear_points_with_noise(
                                                                x_range=house_areas_range_val,
                                                                intercept=50,
                                                                slope=1,
                                                                num_points=10,
                                                                noise_std_dev=30)


# Unpack the returned values
# house_areas, housing_prices = housing_prices

# Plot the points
plt.figure(figsize=(10, 6))
plt.scatter(house_areas, housing_prices, label='House price', color='blue')
plt.xlabel('House Area (sq ft)')
plt.ylabel('Housing Price')
plt.title('Housing Prices vs. House Area with Noise (TRAIN)')
plt.legend()
plt.grid(True)
plt.show()


# Plot the points
plt.figure(figsize=(10, 6))
plt.scatter(house_areas_val, housing_prices_val, label='House price', color='orange')
plt.xlabel('House Area (sq ft)')
plt.ylabel('Housing Price')
plt.title('Housing Prices vs. House Area with Noise (VAL)')
plt.legend()
plt.grid(True)
plt.show()


# normalise inputs and outputs to ensure gradient numerical stability
house_areas_normalised = house_areas / 1000
housing_prices_normalised = housing_prices / 1000

house_areas_normalised = np.reshape(house_areas_normalised, (-1, 1, 1))
housing_prices_normalised = np.reshape(housing_prices_normalised, (-1, 1, 1))
print(f'{house_areas_normalised.shape=}')
print(f'{housing_prices_normalised.shape=}')


house_areas_normalised = house_areas_normalised
housing_prices_normalised = housing_prices_normalised

# model = Sequential([
#     Linear(in_features=1, out_features=1),
#     Tanh()
#                     ])

model = Linear(in_features=1, out_features=1)

loss_func = MSE()

learning_rate = 0.01
num_epochs = 150
losses = []

for epoch in range(num_epochs):
    loss = 0
    for x, y in zip(house_areas_normalised, housing_prices_normalised):

        y_pred = model.forward(x)
        loss += loss_func.forward(y, y_pred)
        error_grads = loss_func.backward(y, y_pred)
        model.backward(error_grads, learning_rate)

    loss /= len(house_areas_normalised)
    print(loss)
    losses.append(loss)

plt.figure(figsize=(10, 6))
plt.plot(range(len(losses)), losses, label='Train loss')
plt.xlabel('House Area (sq ft)')
plt.ylabel('Housing Price')
plt.title('Housing Prices vs. House Area with Noise')
plt.legend()
plt.grid(True)
plt.show()
