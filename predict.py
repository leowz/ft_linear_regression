#!/usr/bin/env python3
"""
Prediction program for ft_linear_regression.
Predicts the price of a car based on its mileage using trained theta values.
"""

import os

THETA_FILE = "theta.csv"


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
    theta0, theta1 = load_theta()

    if theta0 == 0.0 and theta1 == 0.0:
        print("Warning: Model has not been trained yet. Using theta0=0 and theta1=0.")

    try:
        mileage_input = input("Enter the mileage (km): ")
        mileage = float(mileage_input)

        if mileage < 0:
            print("Error: Mileage cannot be negative.")
            return

        price = estimate_price(mileage, theta0, theta1)

        if price < 0:
            print(f"Estimated price for {mileage:.0f} km: 0 (model predicts negative value)")
        else:
            print(f"Estimated price for {mileage:.0f} km: {price:.2f}")

    except ValueError:
        print("Error: Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
