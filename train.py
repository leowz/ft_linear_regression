#!/usr/bin/env python3
"""
Training program for ft_linear_regression.
Trains a linear regression model using gradient descent algorithm.
"""

DATA_FILE = "data.csv"
THETA_FILE = "theta.csv"
LEARNING_RATE = 0.1
ITERATIONS = 1000


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


def train(mileages, prices, learning_rate, iterations):
    """Train the model using gradient descent."""
    # Normalize the data for stable gradient descent
    norm_mileages, min_km, max_km = normalize(mileages)
    norm_prices, min_price, max_price = normalize(prices)

    theta0 = 0.0
    theta1 = 0.0
    m = len(mileages)

    for i in range(iterations):
        # Calculate gradients using the specified formulas
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

    # Denormalize theta values to work with original data
    price_range = max_price - min_price
    km_range = max_km - min_km

    theta1_real = theta1 * price_range / km_range
    theta0_real = theta0 * price_range + min_price - theta1_real * min_km

    return theta0_real, theta1_real


def save_theta(theta0, theta1):
    """Save theta values to CSV file."""
    with open(THETA_FILE, 'w') as f:
        f.write("theta0,theta1\n")
        f.write(f"{theta0},{theta1}\n")


def main():
    print("Loading data...")
    mileages, prices = load_data(DATA_FILE)
    print(f"Loaded {len(mileages)} data points.")

    print(f"Training model (learning_rate={LEARNING_RATE}, iterations={ITERATIONS})...")
    theta0, theta1 = train(mileages, prices, LEARNING_RATE, ITERATIONS)

    print(f"Training complete!")
    print(f"  theta0 (intercept): {theta0:.6f}")
    print(f"  theta1 (slope):     {theta1:.6f}")

    save_theta(theta0, theta1)
    print(f"Model saved to {THETA_FILE}")


if __name__ == "__main__":
    main()
