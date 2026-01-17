#!/usr/bin/env python3
"""
Visualize how learning rate and iterations affect convergence.
Plots the cost function (MSE) over iterations for different learning rates.
"""

import matplotlib.pyplot as plt

DATA_FILE = "data.csv"


def load_data(filename):
    """Load mileage and price data from CSV file."""
    mileages = []
    prices = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) == 2:
                mileages.append(float(parts[0]))
                prices.append(float(parts[1]))

    return mileages, prices


def normalize(values):
    """Normalize values using min-max normalization to range [0, 1]."""
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.0] * len(values), min_val, max_val
    normalized = [(v - min_val) / (max_val - min_val) for v in values]
    return normalized, min_val, max_val


def estimate_price(mileage, theta0, theta1):
    """Calculate estimated price using the linear hypothesis."""
    return theta0 + (theta1 * mileage)


def calculate_cost(mileages, prices, theta0, theta1):
    """Calculate Mean Squared Error (MSE)."""
    m = len(mileages)
    total_error = 0.0
    for i in range(m):
        prediction = estimate_price(mileages[i], theta0, theta1)
        total_error += (prediction - prices[i]) ** 2
    return total_error / (2 * m)


def train_with_history(mileages, prices, learning_rate, iterations):
    """Train and return cost history for each iteration."""
    norm_mileages, _, _ = normalize(mileages)
    norm_prices, _, _ = normalize(prices)

    theta0 = 0.0
    theta1 = 0.0
    m = len(mileages)
    cost_history = []

    for i in range(iterations):
        # Record cost
        cost = calculate_cost(norm_mileages, norm_prices, theta0, theta1)
        cost_history.append(cost)

        # Calculate gradients
        sum_error = 0.0
        sum_error_mileage = 0.0

        for j in range(m):
            prediction = estimate_price(norm_mileages[j], theta0, theta1)
            error = prediction - norm_prices[j]
            sum_error += error
            sum_error_mileage += error * norm_mileages[j]

        # Update theta0 and theta1 simultaneously
        tmp_theta0 = learning_rate * (1 / m) * sum_error
        tmp_theta1 = learning_rate * (1 / m) * sum_error_mileage

        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    return cost_history


def main():
    mileages, prices = load_data(DATA_FILE)

    # Test different learning rates
    learning_rates = [0.1, 0.5, 1.0, 2.0]
    iterations = 500

    plt.figure(figsize=(12, 5))

    # Plot 1: Different learning rates
    plt.subplot(1, 2, 1)
    for lr in learning_rates:
        cost_history = train_with_history(mileages, prices, lr, iterations)
        # Cap values for display (lr=2.0 will diverge)
        cost_history = [min(c, 1.0) for c in cost_history]
        plt.plot(cost_history, label=f'lr = {lr}')

    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.title('Cost vs Iterations (Different Learning Rates)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Compare lr = 0.1, 0.5, 1.0, 2.0
    plt.subplot(1, 2, 2)
    iterations_long = 500
    for lr in [0.1, 0.5, 1.0, 2.0]:
        cost_history = train_with_history(mileages, prices, lr, iterations_long)
        # Cap values for display (lr=2.0 will diverge)
        cost_history = [min(c, 1.0) for c in cost_history]
        plt.plot(cost_history, label=f'lr = {lr}')

    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.title('Cost vs Iterations (lr = 0.1, 0.5, 1.0, 2.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
