#!/usr/bin/env python3
"""
Visualization program for ft_linear_regression (Bonus).
Plots the data points and the regression line.
"""

import os
import matplotlib.pyplot as plt

DATA_FILE = "data.csv"
THETA_FILE = "theta.csv"


def load_data(filename):
    """Load mileage and price data from CSV file."""
    mileages = []
    prices = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) == 2:
                mileages.append(float(parts[0]))
                prices.append(float(parts[1]))

    return mileages, prices


def load_theta():
    """Load theta0 and theta1 from file. Returns (0, 0) if file doesn't exist."""
    if not os.path.exists(THETA_FILE):
        return 0.0, 0.0

    with open(THETA_FILE, 'r') as f:
        lines = f.readlines()
        if len(lines) < 2:
            return 0.0, 0.0
        theta0 = float(lines[1].split(',')[0])
        theta1 = float(lines[1].split(',')[1])
    return theta0, theta1


def estimate_price(mileage, theta0, theta1):
    """Calculate estimated price using the linear hypothesis."""
    return theta0 + (theta1 * mileage)


def main():
    # Load data
    mileages, prices = load_data(DATA_FILE)
    theta0, theta1 = load_theta()

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot data points
    plt.scatter(mileages, prices, color='blue', label='Data points', alpha=0.7)

    # Plot regression line if model is trained
    if theta0 != 0.0 or theta1 != 0.0:
        min_mileage = min(mileages)
        max_mileage = max(mileages)
        x_line = [min_mileage, max_mileage]
        y_line = [estimate_price(x, theta0, theta1) for x in x_line]
        plt.plot(x_line, y_line, color='red', linewidth=2, label='Regression line')
    else:
        print("Warning: Model not trained yet. Only showing data points.")

    # Labels and title
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.title('Car Price vs Mileage - Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
