# ft_linear_regression

An introduction to machine learning through simple linear regression.

## Table of Contents

1. [What is Linear Regression?](#what-is-linear-regression)
2. [The Math Behind It](#the-math-behind-it)
3. [Gradient Descent Explained](#gradient-descent-explained)
4. [Project Overview](#project-overview)
5. [Installation](#installation)
6. [Usage](#usage)

---

## What is Linear Regression?

Imagine you're trying to predict the price of a used car based on how many kilometers it has been driven. You notice a pattern: cars with more kilometers tend to be cheaper. Linear regression is a way to mathematically describe this relationship.

### The Basic Idea

Linear regression finds the **best straight line** that fits through your data points. Once you have this line, you can use it to make predictions for new data.

Think of it like this:
- You have several data points (each car's mileage and its price)
- You want to draw a line through them that best represents the relationship
- Once you have that line, you can predict the price of any car just by knowing its mileage

### Why "Linear"?

The word "linear" comes from "line". We're assuming the relationship between mileage and price can be represented by a straight line. The equation for a line is:

```
y = mx + b
```

Where:
- `y` is what we want to predict (the price)
- `x` is what we know (the mileage)
- `m` is the slope of the line (how steep it is)
- `b` is the y-intercept (where the line crosses the y-axis)

In machine learning terminology, we often write this as:

```
estimatePrice(mileage) = theta0 + (theta1 * mileage)
```

Where:
- `theta0` (also written as `θ₀`) is the y-intercept
- `theta1` (also written as `θ₁`) is the slope

---

## The Math Behind It

### The Hypothesis Function

Our hypothesis (prediction function) is:

```
h(x) = θ₀ + θ₁ * x
```

This is just the equation of a line where:
- `θ₀` is the starting point (when mileage = 0, this would be the base price)
- `θ₁` is how much the price changes for each additional kilometer

For car prices, `θ₁` will be negative because more kilometers = lower price.

### The Cost Function

How do we know if our line is good? We measure the **error** - how far off our predictions are from the actual values.

For each data point, the error is:
```
error = predicted_price - actual_price
```

We square these errors (to make them all positive and penalize big errors more), then average them. This is called the **Mean Squared Error (MSE)**:

```
MSE = (1/m) * Σ(h(xᵢ) - yᵢ)²
```

Where:
- `m` is the number of data points
- `h(xᵢ)` is our prediction for data point i
- `yᵢ` is the actual value for data point i
- `Σ` means "sum up all of these"

Our goal is to find `θ₀` and `θ₁` that minimize this error.

---

## Gradient Descent Explained

### The Concept

Imagine you're standing on a hill in complete fog. You want to find the lowest point (the valley). What would you do? You'd feel the ground around you and take a step in the direction that goes downhill. Repeat this until you can't go any lower.

This is exactly what gradient descent does! The "hill" is our cost function, and we want to find the values of `θ₀` and `θ₁` that give us the lowest error.

### The Algorithm

The gradient tells us the direction of steepest increase. We want to go the opposite way (decrease), so we subtract the gradient:

```
θ₀ := θ₀ - learning_rate * (∂/∂θ₀)Cost
θ₁ := θ₁ - learning_rate * (∂/∂θ₁)Cost
```

After doing the calculus (taking partial derivatives), we get:

```
tmpθ₀ = learning_rate * (1/m) * Σ(estimatePrice(mileage[i]) - price[i])
tmpθ₁ = learning_rate * (1/m) * Σ(estimatePrice(mileage[i]) - price[i]) * mileage[i]

θ₀ := θ₀ - tmpθ₀
θ₁ := θ₁ - tmpθ₁
```

### Key Terms

**Learning Rate**: This controls how big each step is.
- Too big = you might overshoot the minimum and bounce around
- Too small = it takes forever to converge
- Typical values: 0.001 to 0.1

**Iterations**: How many times we repeat the gradient descent steps. More iterations = more accurate (up to a point).

**Simultaneous Update**: We must calculate BOTH `tmpθ₀` and `tmpθ₁` using the current values of `θ₀` and `θ₁`, THEN update both at the same time. If we update `θ₀` first, we'd use the wrong value when calculating `tmpθ₁`.

### Feature Normalization

When your features (like mileage in kilometers) have very large values (e.g., 150,000 km), gradient descent can be very slow or unstable.

We solve this by **normalizing** the data to a range of [0, 1]:

```
normalized_value = (value - min) / (max - min)
```

After training, we convert the theta values back to work with the original scale.

---

## Project Overview

This project implements a simple linear regression to predict car prices based on mileage.

### Files

| File | Description |
|------|-------------|
| `train.py` | Trains the model using gradient descent and saves theta values |
| `predict.py` | Loads trained theta values and predicts prices for given mileage |
| `visualize.py` | Plots the data points and regression line (bonus) |
| `data.csv` | Dataset with car mileages and prices |
| `theta.csv` | Saved theta values (created after training) |

### How It Works

1. **Training Phase** (`train.py`):
   - Reads the dataset from `data.csv`
   - Normalizes the data for stable gradient descent
   - Runs gradient descent for many iterations
   - Saves the final `θ₀` and `θ₁` values to `theta.csv`

2. **Prediction Phase** (`predict.py`):
   - Loads `θ₀` and `θ₁` from `theta.csv` (defaults to 0 if not trained)
   - Asks the user for a mileage
   - Calculates: `price = θ₀ + θ₁ * mileage`
   - Returns the estimated price

3. **Visualization** (`visualize.py`):
   - Plots all data points as blue dots
   - Draws the regression line in red
   - Shows the relationship between mileage and price

---

## Installation

### Requirements

- Python 3.6+
- matplotlib (for visualization)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ft_linear_regression
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Train the Model

```bash
python train.py
```

Output:
```
Loading data...
Loaded 20 data points.
Training model (learning_rate=0.1, iterations=1000)...
Training complete!
  theta0 (intercept): 8499.599651
  theta1 (slope):     -0.021448
Model saved to theta.csv
```

### Step 2: Predict a Price

```bash
python predict.py
```

```
Enter the mileage (km): 100000
Estimated price for 100000 km: 6354.78
```

### Step 3: Visualize the Results (Bonus)

```bash
python visualize.py
```

This will open a window showing:
- Blue dots: actual data points from the dataset
- Red line: the regression line your model learned

---

## Understanding the Results

After training on our car dataset:
- `θ₀ ≈ 8500`: This represents the base price (theoretical price when mileage = 0)
- `θ₁ ≈ -0.02`: For every additional kilometer, the price decreases by about 0.02 units

This makes sense! Cars with more mileage are worth less, which is reflected in the negative slope.

### Example Predictions

| Mileage | Estimated Price |
|---------|-----------------|
| 0 km | ~8500 |
| 50,000 km | ~7427 |
| 100,000 km | ~6355 |
| 150,000 km | ~5282 |
| 200,000 km | ~4210 |

---

## Summary

Linear regression is one of the simplest machine learning algorithms, but it's incredibly powerful for understanding relationships in data. The key concepts are:

1. **Model**: A line defined by `y = θ₀ + θ₁ * x`
2. **Training**: Finding the best `θ₀` and `θ₁` using gradient descent
3. **Prediction**: Plugging new values into the trained model

This project demonstrates these fundamentals using a practical example of predicting car prices based on mileage.
