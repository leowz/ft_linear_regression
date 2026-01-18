# ft_linear_regression

An introduction to machine learning through simple linear regression.

## Table of Contents

1. [What is Linear Regression?](#what-is-linear-regression)
2. [Key Definitions](#key-definitions)
   - Theta0, Theta1, Cost (MSE), Convergence
3. [The Math Behind It](#the-math-behind-it)
   - The Calculus: Where the formulas come from
   - Theta vs Gradient
   - How Learning Rate affects Step Size
4. [Gradient Descent Explained](#gradient-descent-explained)
5. [Choosing Learning Rate and Iterations](#choosing-learning-rate-and-iterations)
6. [Why Normalization?](#why-normalization)
7. [Project Overview](#project-overview)
8. [Installation](#installation)
9. [Usage](#usage)
   - Includes: plot_convergence.py detailed explanation

---

## What is Linear Regression?

### The Problem We're Solving

Imagine you want to buy a used car. You notice that:
- A car with 50,000 km costs around 7,500
- A car with 100,000 km costs around 6,500
- A car with 150,000 km costs around 5,500

You see a pattern: **more kilometers = lower price**. But how do you predict the price of a car with 80,000 km? Or 200,000 km?

Linear regression answers this question by finding the mathematical relationship between mileage and price.

### The Basic Idea

Linear regression finds the **best straight line** that fits through your data points.

```
Price
  ^
  |  *
  |    *
  |      *
  |        *
  |          *
  +-------------> Mileage
```

Once you have this line, you can:
1. Put in any mileage value
2. Follow the line to find the corresponding price
3. That's your prediction!

### Why "Linear"?

"Linear" means "line". We assume the relationship is a straight line, not a curve.

The equation for any straight line is:

```
y = mx + b
```

You might remember this from school:
- **y** = the vertical position (in our case: price)
- **x** = the horizontal position (in our case: mileage)
- **m** = the slope (how steep the line is)
- **b** = the y-intercept (where the line crosses the y-axis)

In machine learning, we use different names:

```
estimatePrice(mileage) = theta0 + (theta1 * mileage)
```

- **theta0 (θ₀)** = same as "b" (the y-intercept)
- **theta1 (θ₁)** = same as "m" (the slope)

---

## Key Definitions

### Theta0 (θ₀) - The Y-Intercept

**What it is:** Where the line crosses the y-axis (the vertical axis).

**In simple terms:** The price when mileage equals zero.

**Visually:**
```
Price
  ^
  |
8481 + - - - * <- This is theta0 (where line hits the y-axis)
  |           \
  |            \
  |             \
  +--+--+--+--+---> Mileage
  0
```

**Example:** If θ₀ = 8481, it means:
- A car with 0 km would theoretically cost 8481
- This is the "starting price" before mileage reduces it

### Theta1 (θ₁) - The Slope

**What it is:** How steep the line is. How much y changes when x changes by 1.

**In simple terms:** How much the price drops for each additional kilometer.

**Visually:**
```
If theta1 = -0.021:

For every 1 km increase in mileage,
the price drops by 0.021

         Price drops by 0.021
              |
              v
        *─────
              \
               *─────
                     \
                      *
```

**Example:** If θ₁ = -0.021:
- Each kilometer reduces the price by 0.021
- 1,000 km reduces price by: 1,000 × 0.021 = 21
- 100,000 km reduces price by: 100,000 × 0.021 = 2,100

**Why negative?** Because more mileage = lower price. The line goes DOWN as you move RIGHT.

### Putting It Together: The Model

After training, our model is:

```
price = theta0 + (theta1 × mileage)
price = 8481 + (-0.021 × mileage)
```

**Let's calculate some predictions:**

| Mileage | Calculation | Price |
|---------|-------------|-------|
| 0 km | 8481 + (-0.021 × 0) = 8481 + 0 | **8481** |
| 50,000 km | 8481 + (-0.021 × 50000) = 8481 - 1050 | **7431** |
| 100,000 km | 8481 + (-0.021 × 100000) = 8481 - 2100 | **6381** |
| 150,000 km | 8481 + (-0.021 × 150000) = 8481 - 3150 | **5331** |
| 200,000 km | 8481 + (-0.021 × 200000) = 8481 - 4200 | **4281** |

### Cost (Mean Squared Error - MSE)

**What it is:** A number that tells us how wrong our predictions are.

**The idea:**
1. Make a prediction for each data point
2. Calculate the error (prediction - actual)
3. Square each error (to make negatives positive)
4. Average all squared errors

**Why do we need this?**

Imagine two different lines trying to fit the same data:

```
Line A (bad fit):          Line B (good fit):

  *                          *
    \  *                       *
      \    *                     *
        \      *                   *
          \                          *
```

Line B is closer to the points, so it has a **lower cost**.
Line A is far from the points, so it has a **higher cost**.

**The cost tells us how good our line is. Lower = better.**

**Formula:**
```
Cost = (1/2m) × Σ(prediction - actual)²
```

Let's break this down:
- `m` = number of data points (e.g., 24 cars)
- `Σ` = sum (add up all the values)
- `prediction - actual` = the error for one point
- `²` = squared (multiply by itself)
- `1/2m` = divide by 2×m to get the average (the 1/2 is for mathematical convenience)

**Example with 3 cars:**

| Car | Mileage | Actual Price | Our Prediction | Error | Error² |
|-----|---------|--------------|----------------|-------|--------|
| 1 | 100,000 | 6,500 | 6,381 | 6381-6500 = -119 | 14,161 |
| 2 | 150,000 | 5,200 | 5,331 | 5331-5200 = 131 | 17,161 |
| 3 | 200,000 | 4,000 | 4,281 | 4281-4000 = 281 | 78,961 |

```
Cost = (1 / (2×3)) × (14161 + 17161 + 78961)
     = (1/6) × 110283
     = 18,380.5
```

**Why squared?**

1. **Makes all errors positive:**
   - Error of -119 becomes 14,161
   - Error of +131 becomes 17,161
   - Without squaring, -119 and +119 would cancel out!

2. **Punishes big errors more:**
   - Error of 10 → 10² = 100
   - Error of 100 → 100² = 10,000
   - A 10x bigger error becomes 100x worse in the cost

**How Cost Relates to Gradient Descent:**

The entire purpose of gradient descent is to **minimize the cost**.

```
Training Process:

Iteration 1:   Cost = 0.5      (very wrong)
Iteration 10:  Cost = 0.3      (getting better)
Iteration 100: Cost = 0.1      (much better)
Iteration 500: Cost = 0.02     (almost perfect)
Iteration 1000: Cost = 0.015   (converged, can't improve much more)
```

**Key insight:** When the cost stops decreasing, we've found the best theta values.

**What different cost values mean:**

| Cost Value | Interpretation |
|------------|----------------|
| Very high (> 1.0) | Model is completely wrong |
| High (0.1 - 1.0) | Still learning |
| Low (0.01 - 0.1) | Getting close |
| Very low (< 0.01) | Good fit! |
| Zero | Perfect fit (rare in real data) |

*Note: These ranges are for normalized data. Raw data would have much larger cost values.*

### Convergence

**What it means:** The algorithm has finished learning.

**Analogy:** Imagine adjusting the volume on your TV:
- At first, you make big adjustments (way too loud → much quieter)
- Then smaller adjustments (a bit too quiet → slightly louder)
- Eventually, you stop because it sounds right

When the adjustments become tiny (< 0.0001), we say the algorithm has **converged**.

**How to detect convergence:**
- The cost stops decreasing
- Theta values stop changing
- The change between iterations is extremely small

---

## The Math Behind It

### The Hypothesis Function

This is our prediction formula:

```
h(x) = θ₀ + θ₁ × x
```

Or written out:

```
estimatePrice(mileage) = theta0 + (theta1 × mileage)
```

**What each part means:**

| Symbol | Name | Meaning | Example |
|--------|------|---------|---------|
| h(x) | hypothesis | our prediction | predicted price |
| x | input | what we know | mileage (e.g., 100,000) |
| θ₀ | theta0 | y-intercept | 8481 |
| θ₁ | theta1 | slope | -0.021 |

**The goal of training:** Find the best values for θ₀ and θ₁.

### The Training Formulas

These formulas tell us how to adjust theta0 and theta1 to make our predictions better:

```
tmpθ₀ = learningRate × (1/m) × Σ(estimatePrice(mileage[i]) - price[i])

tmpθ₁ = learningRate × (1/m) × Σ(estimatePrice(mileage[i]) - price[i]) × mileage[i]
```

### Where Do These Formulas Come From? (The Calculus)

These formulas come from **calculus** - specifically, taking the **partial derivatives** of the cost function.

**The Cost Function:**
```
J(θ₀, θ₁) = (1/2m) × Σ(θ₀ + θ₁×x - y)²
```

Where:
- J = the cost (what we want to minimize)
- θ₀, θ₁ = our parameters
- x = mileage
- y = actual price

**Taking Partial Derivatives:**

To find the minimum of the cost function, we take the derivative with respect to each parameter:

```
∂J/∂θ₀ = (1/m) × Σ(θ₀ + θ₁×x - y)
       = (1/m) × Σ(prediction - actual)
       = (1/m) × Σ(error)

∂J/∂θ₁ = (1/m) × Σ(θ₀ + θ₁×x - y) × x
       = (1/m) × Σ(error) × x
```

**These partial derivatives ARE the gradients!**

**The Gradient Vector:**
```
∇J = [ ∂J/∂θ₀ ]  =  [ (1/m) × Σ(error)     ]
     [ ∂J/∂θ₁ ]     [ (1/m) × Σ(error × x) ]
```

The gradient points in the direction of **steepest increase** of the cost. We want to **decrease** the cost, so we go the **opposite direction** (subtract).

### Theta vs Gradient - Don't Confuse Them!

| Term | What It Is | Role |
|------|------------|------|
| θ₀, θ₁ | Parameters | The values we're trying to find |
| ∂J/∂θ₀, ∂J/∂θ₁ | Gradients | Direction to adjust parameters |
| tmpθ₀, tmpθ₁ | Gradient × learning rate | Actual step size |

**Analogy:**
- **θ (theta)** = your position on a hill
- **Gradient** = which direction is uphill
- **-Gradient** = which direction is downhill (where we want to go)
- **Learning rate × Gradient** = how big of a step to take

### How Learning Rate Affects Step Size

The learning rate **multiplies** the gradient to determine step size:

```
tmpθ₀ = learningRate × gradient
tmpθ₀ = learningRate × (1/m) × Σ(error)
```

**Example with gradient = 0.5:**

| Learning Rate | Calculation | Step Size | Effect |
|---------------|-------------|-----------|--------|
| 0.1 | 0.1 × 0.5 | 0.05 | Small step (slow, safe) |
| 0.5 | 0.5 × 0.5 | 0.25 | Medium step |
| 1.0 | 1.0 × 0.5 | 0.50 | Large step (fast) |
| 2.0 | 2.0 × 0.5 | 1.00 | Too large (overshoots!) |

**Key insight:** Bigger learning rate = bigger steps

```
Small lr (0.1):     *....*....*....*....* (many small steps)

Large lr (1.0):     *____*____*           (few big steps)

Too large lr (2.0): *         *         * (overshoots, bounces)
                         ←→       ←→
```

### Breaking Down the Formula Piece by Piece

**This looks scary! Let's break it down:**

#### Part 1: The Error

```
estimatePrice(mileage[i]) - price[i]
```

This is just: `prediction - actual = error`

For car #1 with 100,000 km and actual price 6,500:
- prediction = 8481 + (-0.021 × 100000) = 6,381
- actual = 6,500
- error = 6,381 - 6,500 = -119

#### Part 2: The Sum (Σ)

```
Σ(error)
```

The Σ symbol means "add up all of these".

For all 24 cars, we:
1. Calculate the error for car 1
2. Calculate the error for car 2
3. ... continue for all cars ...
4. Add all 24 errors together

#### Part 3: The Average

```
(1/m) × Σ(error)
```

Dividing by m (number of cars) gives us the **average error**.

If m = 24 and total error = 240, then average = 240/24 = 10

#### Part 4: The Learning Rate

```
learningRate × average_error
```

The learning rate (e.g., 0.1) controls how big our adjustment is.

- Big learning rate (1.0) = big adjustments = fast but risky
- Small learning rate (0.01) = tiny adjustments = slow but safe

#### Part 5: Why tmpθ₁ has an extra × mileage[i]

For theta1 (the slope), we multiply by the mileage:

```
Σ(error × mileage[i])
```

**Why?** Because theta1 is multiplied by mileage in the formula. When we figure out how to adjust theta1, we need to account for that multiplication.

**Intuition:** If a car has high mileage and we predicted wrong, theta1 (which affects high-mileage cars more) needs more adjustment.

#### The Complete Process

```python
# For each iteration:

# Step 1: Calculate all errors
errors = []
for each car:
    prediction = theta0 + (theta1 × mileage)
    error = prediction - actual_price
    errors.append(error)

# Step 2: Calculate adjustments
sum_of_errors = sum(all errors)
sum_of_errors_times_mileage = sum(error × mileage for each car)

tmp_theta0 = learning_rate × (1/m) × sum_of_errors
tmp_theta1 = learning_rate × (1/m) × sum_of_errors_times_mileage

# Step 3: Update theta values (SIMULTANEOUSLY!)
theta0 = theta0 - tmp_theta0
theta1 = theta1 - tmp_theta1
```

#### Why Simultaneous Update?

We must update both thetas at the same time. Here's why:

**Wrong way:**
```python
theta0 = theta0 - tmp_theta0  # theta0 changed!
# Now when we calculate tmp_theta1, we use the NEW theta0
# This messes up the calculation!
tmp_theta1 = ... uses new theta0 ...  # WRONG!
theta1 = theta1 - tmp_theta1
```

**Right way:**
```python
# Calculate both adjustments using current values
tmp_theta0 = ... uses current theta0 and theta1 ...
tmp_theta1 = ... uses current theta0 and theta1 ...

# Then update both
theta0 = theta0 - tmp_theta0
theta1 = theta1 - tmp_theta1
```

---

## Gradient Descent Explained

### The Core Equations (Mathematical Foundation)

**Equation 1: The Hypothesis (Prediction Function)**
```
h(x) = θ₀ + θ₁·x
```

**Equation 2: The Cost Function (Mean Squared Error)**
```
J(θ₀, θ₁) = (1/2m) × Σᵢ₌₁ᵐ (h(xᵢ) - yᵢ)²
```

Expanded form:
```
J(θ₀, θ₁) = (1/2m) × Σᵢ₌₁ᵐ (θ₀ + θ₁·xᵢ - yᵢ)²
```

**Equation 3: The Gradient (Partial Derivatives)**
```
∂J/∂θ₀ = (1/m) × Σᵢ₌₁ᵐ (h(xᵢ) - yᵢ)

∂J/∂θ₁ = (1/m) × Σᵢ₌₁ᵐ (h(xᵢ) - yᵢ)·xᵢ
```

**Equation 4: The Update Rule**
```
θ₀ := θ₀ - α × ∂J/∂θ₀
θ₁ := θ₁ - α × ∂J/∂θ₁
```

Or in vector notation:
```
θ := θ - α × ∇J(θ)
```

Where:
| Symbol | Name | Mathematical Meaning |
|--------|------|---------------------|
| θ₀, θ₁ | Parameters | Values we optimize |
| α | Learning rate | Step size, α ∈ (0, 2) |
| J(θ) | Cost function | Objective to minimize |
| ∇J(θ) | Gradient | Vector of partial derivatives |
| m | Sample size | Number of training examples |
| h(x) | Hypothesis | Our prediction function |
| := | Assignment | Update the value |

---

### Deriving the Gradient (Step-by-Step Calculus)

This section shows **exactly** how we derive the gradient formulas from the cost function.

**Starting Point: The Cost Function**
```
J(θ₀, θ₁) = (1/2m) × Σᵢ₌₁ᵐ (θ₀ + θ₁·xᵢ - yᵢ)²
```

**Deriving ∂J/∂θ₀:**

Using the chain rule: ∂/∂θ₀ of (something)² = 2·(something)·∂(something)/∂θ₀

```
∂J/∂θ₀ = (1/2m) × Σᵢ₌₁ᵐ 2·(θ₀ + θ₁·xᵢ - yᵢ)·∂(θ₀ + θ₁·xᵢ - yᵢ)/∂θ₀
```

Since ∂(θ₀ + θ₁·xᵢ - yᵢ)/∂θ₀ = 1:
```
∂J/∂θ₀ = (1/2m) × Σᵢ₌₁ᵐ 2·(θ₀ + θ₁·xᵢ - yᵢ)·1
       = (1/m) × Σᵢ₌₁ᵐ (θ₀ + θ₁·xᵢ - yᵢ)
       = (1/m) × Σᵢ₌₁ᵐ (h(xᵢ) - yᵢ)
       = (1/m) × Σᵢ₌₁ᵐ errorᵢ
```

**Deriving ∂J/∂θ₁:**

```
∂J/∂θ₁ = (1/2m) × Σᵢ₌₁ᵐ 2·(θ₀ + θ₁·xᵢ - yᵢ)·∂(θ₀ + θ₁·xᵢ - yᵢ)/∂θ₁
```

Since ∂(θ₀ + θ₁·xᵢ - yᵢ)/∂θ₁ = xᵢ:
```
∂J/∂θ₁ = (1/2m) × Σᵢ₌₁ᵐ 2·(θ₀ + θ₁·xᵢ - yᵢ)·xᵢ
       = (1/m) × Σᵢ₌₁ᵐ (θ₀ + θ₁·xᵢ - yᵢ)·xᵢ
       = (1/m) × Σᵢ₌₁ᵐ (h(xᵢ) - yᵢ)·xᵢ
       = (1/m) × Σᵢ₌₁ᵐ errorᵢ·xᵢ
```

**Summary of Derivation:**
```
∂J/∂θ₀ = (1/m) × Σ(error)           ← average error
∂J/∂θ₁ = (1/m) × Σ(error × x)       ← weighted average error
```

**The Gradient Vector:**
```
∇J(θ) = [ ∂J/∂θ₀ ]  =  [ (1/m) × Σ(h(x) - y)     ]
        [ ∂J/∂θ₁ ]     [ (1/m) × Σ(h(x) - y)·x   ]
```

---

### The Algorithm (Implementation)

**The gradient descent update rule:**

```
repeat until converged:
    θ₀ := θ₀ - α × (1/m) × Σ(h(xᵢ) - yᵢ)
    θ₁ := θ₁ - α × (1/m) × Σ(h(xᵢ) - yᵢ)·xᵢ
```

Or in compact notation:
```
θ := θ - α × ∇J(θ)
```

**In Python:**
```python
for i in range(iterations):
    # Calculate gradients (partial derivatives)
    gradient_theta0 = (1/m) * sum(predictions - actuals)
    gradient_theta1 = (1/m) * sum((predictions - actuals) * mileages)

    # Update parameters (NOTE: subtract, not add!)
    theta0 = theta0 - learning_rate * gradient_theta0
    theta1 = theta1 - learning_rate * gradient_theta1
```

---

### What is Gradient Descent?

**The problem:** We need to find the best θ₀ and θ₁, but there are infinite possibilities!

**The solution:** Start with a guess and improve it step by step.

**Simple analogy - Finding the lowest point while blindfolded:**

```
Imagine you're on a hilly landscape, blindfolded:

    You are here
         ↓
         *
        /|\
       / | \      1. Feel the slope under your feet
      /  |  \     2. Take a step DOWNHILL
     /   ↓   \    3. Repeat until flat (can't go lower)
    /    *    \   4. You found the bottom!
   /    / \    \
  /    /   \    \
 /____/_____\____\
      ↑
   Bottom (minimum)
```

**In our case:**
| Analogy | Linear Regression |
|---------|-------------------|
| Landscape | Cost function J(θ) |
| Your position | Current θ₀, θ₁ values |
| Height | How wrong predictions are (cost) |
| Slope/steepness | The gradient |
| Bottom of valley | Best θ₀, θ₁ (minimum cost) |

---

### Mathematical Properties of the Cost Function

**Why Gradient Descent Works: Convexity**

The MSE cost function is **convex**, which means:
1. It has exactly ONE minimum (no local minima traps)
2. Gradient descent is guaranteed to find it
3. Any downhill direction leads to the minimum

**Mathematical Definition of Convexity:**

A function J(θ) is convex if for any two points θ₁ and θ₂:
```
J(t·θ₁ + (1-t)·θ₂) ≤ t·J(θ₁) + (1-t)·J(θ₂)   for all t ∈ [0,1]
```

**Proving MSE is Convex:**

The MSE cost function:
```
J(θ) = (1/2m) × Σᵢ(θ₀ + θ₁·xᵢ - yᵢ)²
```

The Hessian (second derivative matrix) is:
```
H = ∇²J = (1/m) × [ m      Σxᵢ   ]
                  [ Σxᵢ    Σxᵢ²  ]
```

A function is convex if its Hessian is positive semi-definite (all eigenvalues ≥ 0).

For MSE, the Hessian is always positive semi-definite, therefore:
- **MSE is convex**
- Gradient descent will always converge to the global minimum
- There are no local minima to get trapped in

**Visual Consequence:**

```
Non-convex (bad):           Convex (MSE - good):

   /\      /\                     \      /
  /  \    /  \                     \    /
 /    \  /    \                     \  /
/      \/      \                     \/
    ↑                                ↑
Local minima exist!           Only ONE minimum!
Might get stuck              Always find best θ
```

### Visualizing the Cost Function

The cost function looks like a bowl. We want to find the bottom:

```
    Cost J(θ)
      ^
      |    \         /
      |     \       /
      |      \     /
      |       \   /
      |        \_/  ← minimum (best theta)
      |
      +---------------→ θ

Gradient descent "rolls" down to the bottom:

      Start (random guess)
           ↓
           *
            \
             \  ← step 1
              *
               \
                \  ← step 2
                 *
                  \
                   * ← converged (done!)
```

---

### Why Subtract? (The Minus Sign Explained)

You might wonder: **why `θ = θ - gradient` and not `θ = θ + gradient`?**

#### Reason 1: The Gradient Points UPHILL

From calculus:
```
Gradient = direction of steepest INCREASE
```

But we want to DECREASE cost (go downhill), so we go the OPPOSITE direction:
```
-Gradient = direction of steepest DECREASE
```

**Visual:**
```
Cost J(θ)
    ^
    |        * ← you are here
    |       /
    |      /  ↑ gradient points UP (increasing cost)
    |     /   |
    |    /
    |   /
    +--+--------→ θ

To go DOWN (decrease cost), go OPPOSITE of gradient → SUBTRACT
```

#### Reason 2: Mathematical Proof (Taylor Series)

**First-Order Taylor Expansion:**

For a small step d from current position θ:
```
J(θ + d) ≈ J(θ) + ∇J(θ)ᵀ · d
```

Where:
- J(θ + d) = cost at new position
- J(θ) = cost at current position
- ∇J(θ)ᵀ · d = dot product (directional derivative)

**Finding the Optimal Direction:**

We want to find d that **minimizes** J(θ + d):
```
J(θ + d) = J(θ) + ∇J(θ)ᵀ · d
```

Since J(θ) is constant, we need to minimize ∇J(θ)ᵀ · d.

**The Dot Product Analysis:**
```
∇J(θ)ᵀ · d = |∇J(θ)| × |d| × cos(angle)
```

This is minimized when cos(angle) = -1, which means angle = 180°.

**Conclusion:**
```
d should point in the OPPOSITE direction of ∇J(θ)

Therefore: d = -α × ∇J(θ)   (where α > 0 controls step size)

Final update rule: θ_new = θ_old + d = θ_old - α × ∇J(θ)
```

**The minus sign is mathematically derived, not arbitrary!**

#### Reason 3: Formal Derivation

**Goal:** Show that J(θ - α∇J) < J(θ) for small α > 0

**Proof:**
```
Let d = -α × ∇J(θ)

J(θ + d) ≈ J(θ) + ∇J(θ)ᵀ · d
         = J(θ) + ∇J(θ)ᵀ · (-α × ∇J(θ))
         = J(θ) - α × ∇J(θ)ᵀ · ∇J(θ)
         = J(θ) - α × ||∇J(θ)||²
```

Since ||∇J(θ)||² ≥ 0 and α > 0:
```
J(θ + d) = J(θ) - α × ||∇J(θ)||² ≤ J(θ)
```

**The cost decreases (or stays same if gradient = 0)!**

This proves:
- Subtracting gradient always decreases cost (if gradient ≠ 0)
- The decrease is proportional to α × ||∇J||²
- Larger gradient = larger decrease

#### Reason 3: Practical Examples

```
Example 1: Gradient = +0.5 (slope going up)
   θ = θ - (+0.5)
   θ = θ - 0.5     → θ decreases → moves LEFT → cost decreases ✓

Example 2: Gradient = -0.5 (slope going down)
   θ = θ - (-0.5)
   θ = θ + 0.5     → θ increases → moves RIGHT → cost decreases ✓
```

The minus sign automatically handles both directions!

#### Summary: Why Subtract

| Question | Answer |
|----------|--------|
| What does gradient tell us? | Direction of steepest **increase** |
| What do we want? | To **decrease** cost |
| How to decrease? | Go **opposite** of gradient |
| Math operation? | **Subtract** the gradient |
| Why called "descent"? | We **descend** (go down) by subtracting |

```python
# This is why the code uses -= (minus equals):
theta0 -= learning_rate * gradient0  # subtract to go downhill
theta1 -= learning_rate * gradient1  # subtract to go downhill
```

---

### Step-by-Step Walkthrough

Let's trace through gradient descent with actual numbers:

**Setup:**
- Start with θ₀ = 0, θ₁ = 0
- Learning rate α = 0.1
- Data: normalized mileages and prices (0 to 1)

**Iteration 0 (Starting point):**
```
θ₀ = 0, θ₁ = 0
Prediction for any car: price = 0 + 0 × mileage = 0
Actual prices: around 0.5 to 1.0
Error: We're predicting WAY too low!
Gradient: Large negative (need to increase θ₀)
```

**Iteration 1:**
```
gradient_θ₀ = -0.5 (negative = θ₀ too small)
gradient_θ₁ = -0.2

θ₀ = 0 - 0.1 × (-0.5) = 0 + 0.05 = 0.05  ← increased!
θ₁ = 0 - 0.1 × (-0.2) = 0 + 0.02 = 0.02  ← increased!
```

**Iteration 10:**
```
θ₀ ≈ 0.3
θ₁ ≈ 0.05
Predictions getting better, but still off
Gradients smaller now
```

**Iteration 100:**
```
θ₀ ≈ 0.6
θ₁ ≈ -0.3
Getting close!
θ₁ is now negative (correct - more mileage = lower price)
```

**Iteration 500:**
```
θ₀ ≈ 0.85
θ₁ ≈ -0.89
Almost there
Gradients very small
```

**Iteration 1000:**
```
θ₀ ≈ 0.87
θ₁ ≈ -0.95
CONVERGED!
Gradients ≈ 0 (can't improve anymore)
```

**Progress Table:**

| Iteration | θ₀ | θ₁ | Gradient Size | Status |
|-----------|------|-------|---------------|--------|
| 0 | 0 | 0 | Large | Starting |
| 1 | 0.05 | 0.02 | Large | Just started |
| 10 | 0.3 | 0.05 | Medium | Learning |
| 100 | 0.6 | -0.3 | Small | Getting close |
| 500 | 0.85 | -0.89 | Very small | Almost there |
| 1000 | 0.87 | -0.95 | ≈ 0 | **Converged!** |

---

### The Complete Algorithm

```python
def gradient_descent(X, y, learning_rate, iterations):
    theta0 = 0  # Start with initial guess
    theta1 = 0
    m = len(X)  # Number of samples

    for i in range(iterations):
        # Step 1: Make predictions with current theta
        predictions = theta0 + theta1 * X

        # Step 2: Calculate errors
        errors = predictions - y

        # Step 3: Calculate gradients (partial derivatives)
        gradient0 = (1/m) * sum(errors)
        gradient1 = (1/m) * sum(errors * X)

        # Step 4: Update theta (SUBTRACT gradient!)
        theta0 = theta0 - learning_rate * gradient0
        theta1 = theta1 - learning_rate * gradient1

        # Step 5: Check if converged (gradients ≈ 0)
        if abs(gradient0) < 0.0001 and abs(gradient1) < 0.0001:
            print(f"Converged at iteration {i}")
            break

    return theta0, theta1
```

---

### Quick Reference

| Term | Symbol | Meaning |
|------|--------|---------|
| Parameters | θ₀, θ₁ | What we're finding (intercept, slope) |
| Learning rate | α | Step size (typically 0.01 to 1.0) |
| Gradient | ∇J | Direction of steepest increase |
| Cost | J(θ) | How wrong our predictions are |
| Converged | - | Gradients ≈ 0, can't improve |

**The key insight:** Gradient descent finds the minimum by repeatedly taking steps in the **opposite** direction of the gradient (subtracting)

---

### Mathematical Summary (All Equations)

This is a complete reference of all equations used in linear regression with gradient descent:

**1. The Model (Hypothesis):**
```
h(x) = θ₀ + θ₁·x
```

**2. Cost Function (Mean Squared Error):**
```
J(θ₀, θ₁) = (1/2m) × Σᵢ₌₁ᵐ (h(xᵢ) - yᵢ)²
          = (1/2m) × Σᵢ₌₁ᵐ (θ₀ + θ₁·xᵢ - yᵢ)²
```

**3. Gradient (Partial Derivatives):**
```
∂J/∂θ₀ = (1/m) × Σᵢ₌₁ᵐ (h(xᵢ) - yᵢ)

∂J/∂θ₁ = (1/m) × Σᵢ₌₁ᵐ (h(xᵢ) - yᵢ)·xᵢ
```

**4. Gradient Vector:**
```
∇J(θ) = [ ∂J/∂θ₀ ]
        [ ∂J/∂θ₁ ]
```

**5. Update Rule:**
```
θ₀ := θ₀ - α · ∂J/∂θ₀
θ₁ := θ₁ - α · ∂J/∂θ₁
```

Or in vector form:
```
θ := θ - α · ∇J(θ)
```

**6. Normalization:**
```
x_norm = (x - x_min) / (x_max - x_min)
y_norm = (y - y_min) / (y_max - y_min)
```

**7. Denormalization (converting θ back to real scale):**
```
θ₁_real = θ₁_norm × (price_range / km_range)
θ₀_real = θ₀_norm × price_range + min_price - θ₁_real × min_km
```

**8. Convergence Condition:**
```
0 < α < 2/λ_max    (where λ_max = largest eigenvalue of Hessian)

For normalized data: 0 < α < 2
```

**9. Cost Decrease Guarantee:**
```
J(θ - α∇J) ≈ J(θ) - α × ||∇J||²  ≤  J(θ)
```

**10. The Complete Algorithm:**
```
Initialize: θ₀ = 0, θ₁ = 0
Normalize: x, y → x_norm, y_norm

For i = 1 to iterations:
    For j = 1 to m:
        errorⱼ = (θ₀ + θ₁·x_norm[j]) - y_norm[j]

    gradient₀ = (1/m) × Σⱼ errorⱼ
    gradient₁ = (1/m) × Σⱼ errorⱼ · x_norm[j]

    θ₀ := θ₀ - α × gradient₀
    θ₁ := θ₁ - α × gradient₁

Denormalize: θ₀_norm, θ₁_norm → θ₀_real, θ₁_real
```

---

## Choosing Learning Rate and Iterations

### Learning Rate

**What it does:** Controls the size of each step in gradient descent.

**Analogy:** Walking down a hill
- Small learning rate = baby steps (safe but slow)
- Large learning rate = giant leaps (fast but might overshoot)

### The Valid Range

For normalized data (scaled to 0-1):

```
0 < α < 2
```

| Learning Rate (α) | Behavior |
|-------------------|----------|
| 0 | No movement (stuck forever) |
| 0.01 - 0.1 | Very safe, but slow |
| 0.1 - 0.5 | Safe and reasonably fast |
| 0.5 - 1.0 | Fast, still stable |
| 1.0 - 1.9 | Very fast, getting risky |
| ≥ 2.0 | **DIVERGES** (explodes!) |

### Mathematical Reason for the 0 < α < 2 Bound

**The Convergence Condition:**

For gradient descent to converge on a quadratic cost function (like MSE), the learning rate must satisfy:
```
0 < α < 2/λ_max
```

Where λ_max is the largest eigenvalue of the Hessian matrix H = ∇²J(θ).

**For Normalized Data:**

When data is normalized to [0, 1], the Hessian has eigenvalues close to 1:
```
λ_max ≈ 1

Therefore: 0 < α < 2/1 = 2
```

**Why This Bound Exists (Mathematical Intuition):**

Consider a 1D quadratic: J(θ) = (1/2)·λ·θ²

The gradient is: ∇J = λ·θ

The update rule: θ_new = θ - α·λ·θ = θ(1 - αλ)

For convergence, we need |1 - αλ| < 1:
```
-1 < 1 - αλ < 1
-2 < -αλ < 0
0 < αλ < 2
0 < α < 2/λ
```

**At the boundary (α = 2/λ):**
```
θ_new = θ(1 - 2) = -θ
```
The value flips sign each iteration but never converges!

**Beyond the boundary (α > 2/λ):**
```
|1 - αλ| > 1
θ grows exponentially → DIVERGENCE
```

### Why Does lr ≥ 2 Diverge?

**Analogy:** Jumping across a valley

```
With lr = 0.5 (good):

    *                      You take reasonable steps
     \                     and reach the bottom
      \
       *
        \
         *  ← bottom

With lr = 2.0 (bad):

    *                      You jump too far,
     \        *            land on the other side,
      \      /             jump back even further,
       \    /              and bounce forever!
        \  /
         \/
```

When lr = 2, you overshoot by exactly the same distance. When lr > 2, you overshoot even more each time, and the values explode to infinity.

### Visual Comparison

```
lr = 0.1:  \_\_\_\_\_\___     (tiny steps, very slow)

lr = 0.5:  \__\__\___        (medium steps, good speed)

lr = 1.0:  \_  \__  \___     (big steps, fast)

lr = 2.0:  \  /\  /\  /\     (overshooting, bouncing)
            \/  \/  \/
```

### Iterations

**What it is:** How many times we repeat the gradient descent update.

**Rule:** Keep going until converged (values stop changing).

### Iterations Needed for Different Learning Rates

For our dataset:

| Learning Rate | Iterations Needed to Converge |
|---------------|-------------------------------|
| 0.1 | ~1,900 iterations |
| 0.3 | ~630 iterations |
| 0.5 | ~375 iterations |
| 0.7 | ~270 iterations |
| 1.0 | ~185 iterations |
| 1.5 | ~122 iterations |
| 1.7 | ~119 iterations (fastest!) |
| 1.9 | Unstable |
| 2.0 | **Diverges** |

**Observation:** Higher learning rate = fewer iterations needed (until you hit the danger zone).

### How to Choose

**Method 1: Use safe defaults**
- lr = 0.1, iterations = 1000
- Works for most datasets
- Not optimal, but reliable

**Method 2: Experiment and visualize**
1. Run `python plot_convergence.py`
2. Look at the cost curves for different learning rates
3. Pick the highest lr that still converges smoothly
4. Set iterations to where the cost flattens out

**For this dataset:**
- **Safe choice:** lr = 0.1, iterations = 1000
- **Optimized:** lr = 1.0, iterations = 200
- **Maximum speed:** lr = 1.7, iterations = 120

---

## Why Normalization?

### The Problem Without Normalization

Our data has very different scales:
- Mileage: 22,899 to 240,000 (big numbers!)
- Price: 3,650 to 8,290 (smaller numbers)

**The gradient formula for theta1:**
```
tmpθ₁ = learningRate × (1/m) × Σ(error × mileage)
```

**Let's calculate:**
- Error might be around 10,000 (difference between prediction and actual)
- Mileage might be 100,000
- error × mileage = 10,000 × 100,000 = **1,000,000,000** (one billion!)

Even with a tiny learning rate of 0.00000001:
- Adjustment = 0.00000001 × 1,000,000,000 = 10

An adjustment of 10 per iteration is WAY too big. The values will bounce around wildly and never settle. This is called **divergence**.

**Real example - what actually happens without normalization:**

```
Training WITHOUT normalization (learning_rate = 0.0000001):
--------------------------------------------------
Iter 0: theta0=           0.00, theta1=              58.29
Iter 1: theta0=          -0.59, theta1=         -74,922.53
Iter 2: theta0=         756.63, theta1=      96,375,580.03
Iter 3: theta0=     -973275.22, theta1= -123,971,336,620.51
Iter 4: theta0=  1251958534.85, theta1= 159,468,739,960,139.78
Iter 5: theta0=-1610438795589.99, theta1= -205,130,312,519,816,704.00
...
Eventually: theta0 = NaN, theta1 = NaN (exploded to infinity)
```

Even with a **tiny** learning rate of 0.0000001, the values explode astronomically!

**Why does this happen?**

The gradient formula multiplies error by mileage:
```
tmpθ₁ = learning_rate × (1/m) × Σ(error × mileage)
```

With raw data:
- Initial prediction = 0 (theta0=0, theta1=0)
- Actual price ≈ 6,500
- Error = 0 - 6,500 = -6,500
- Mileage ≈ 100,000
- error × mileage = -6,500 × 100,000 = **-650,000,000,000** (billions!)

The gradient is astronomically large, causing wild oscillations that grow exponentially.

### Can We Just Use a Smaller Learning Rate?

You might think: "Just use a tinier learning rate!"

| Learning Rate | Result |
|---------------|--------|
| 0.0000001 | Diverges (as shown above) |
| 0.00000001 | Still diverges |
| 0.000000001 | Still diverges |
| 0.0000000000001 | Might work, but needs millions of iterations |

**It's impractical.** Normalization is the standard solution.

### The Solution: Normalization

**Normalization** scales all values to a range of 0 to 1.

**Formula:**
```
normalized = (value - min) / (max - min)
```

**Example with mileage:**
- Min mileage = 22,899
- Max mileage = 240,000
- Range = 240,000 - 22,899 = 217,101

For a car with 100,000 km:
```
normalized = (100,000 - 22,899) / 217,101 = 0.355
```

**Example with price:**
- Min price = 3,650
- Max price = 8,290
- Range = 8,290 - 3,650 = 4,640

For a price of 6,500:
```
normalized = (6,500 - 3,650) / 4,640 = 0.614
```

**After normalization:**
- All mileages are between 0 and 1
- All prices are between 0 and 1
- Gradients stay small (around 0.1 to 1.0)
- Gradient descent works properly!

### Denormalization - Converting Back to Real Values

After training with normalized data, our theta values only work with normalized inputs:
```
normalized_price = θ₀_norm + θ₁_norm × normalized_mileage
```

But `predict.py` uses **real** mileage (like 100,000 km), so we need to convert theta values back to real scale.

**The Math Derivation:**

Step 1 - Normalization formulas:
```
x_norm = (x - min_km) / km_range
y_norm = (y - min_price) / price_range
```

Step 2 - Our trained model (works with normalized values):
```
y_norm = θ₀_norm + θ₁_norm × x_norm
```

Step 3 - Substitute the normalization formulas:
```
(y - min_price) / price_range = θ₀_norm + θ₁_norm × (x - min_km) / km_range
```

Step 4 - Solve for y (multiply both sides by price_range):
```
y - min_price = θ₀_norm × price_range + θ₁_norm × (price_range / km_range) × (x - min_km)
```

Step 5 - Expand (x - min_km) and add min_price to both sides:
```
y = θ₀_norm × price_range + θ₁_norm × (price_range / km_range) × x
    - θ₁_norm × (price_range / km_range) × min_km + min_price
```

Step 6 - Group terms to match the form y = θ₀_real + θ₁_real × x:

We want: `y = [something] + [something] × x`

Looking at Step 5, identify the coefficient of x and the constant terms:
```
y = [θ₁_norm × (price_range / km_range)] × x      ← this is θ₁_real
    + [θ₀_norm × price_range - θ₁_norm × (price_range / km_range) × min_km + min_price]
                                                   ↑ this is θ₀_real
```

So we define:
```
θ₁_real = θ₁_norm × (price_range / km_range)
```

And for θ₀_real, substitute θ₁_real into the constant terms:
```
θ₀_real = θ₀_norm × price_range - θ₁_real × min_km + min_price
```

Rearranged:
```
θ₀_real = θ₀_norm × price_range + min_price - θ₁_real × min_km
```

**Final Result:** `y = θ₀_real + θ₁_real × x` (works with real mileage!)

**The Code:**
```python
price_range = max_price - min_price
km_range = max_km - min_km

theta1_real = theta1 * price_range / km_range
theta0_real = theta0 * price_range + min_price - theta1_real * min_km
```

**Example:**

| Value | Normalized | Real |
|-------|------------|------|
| θ₀ | 0.94 | 8,481 |
| θ₁ | -1.00 | -0.021 |
| Input (mileage) | 0.355 | 100,000 km |
| Output (price) | 0.614 | 6,354 |

Both calculations give the same result - just in different scales!

### Summary

| Without Normalization | With Normalization |
|-----------------------|---------------------|
| Mileage: 22,899 - 240,000 | Mileage: 0 - 1 |
| Price: 3,650 - 8,290 | Price: 0 - 1 |
| Gradients: billions | Gradients: ~0.1 to 1.0 |
| Result: **DIVERGES** | Result: **Converges** |
| Learning rate: impossible to tune | Learning rate: 0.1 - 1.7 |

**Normalization is not optional for this dataset - it's required for gradient descent to work.**

---

## Project Overview

### Files

| File | Description |
|------|-------------|
| `train.py` | Trains the model using gradient descent |
| `predict.py` | Predicts price for given mileage |
| `visualize.py` | Plots data points and regression line (bonus) |
| `plot_convergence.py` | Visualizes learning rate and iteration effects |
| `data.csv` | Dataset (mileage, price) |
| `theta.csv` | Saved model parameters |
| `requirements.txt` | Python dependencies |

### How the Programs Work Together

```
1. TRAIN
   data.csv → [train.py] → theta.csv
   (raw data)   (learns)    (saves θ₀, θ₁)

2. PREDICT
   theta.csv + user input → [predict.py] → predicted price
   (loads θ₀, θ₁)  (mileage)              (result)

3. VISUALIZE
   data.csv + theta.csv → [visualize.py] → graph
   (points)   (line)       (displays)
```

---

## Installation

### Requirements

- Python 3.6 or higher
- matplotlib (for visualization)

### Setup Steps

```bash
# 1. Navigate to project directory
cd ft_linear_regression

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import matplotlib; print('matplotlib OK')"
```

---

## Usage

### Step 1: Train the Model

```bash
python train.py
```

**Output:**
```
Loading data...
Loaded 24 data points.
Training model (learning_rate=0.1, iterations=1000)...
Training complete!
  theta0 (intercept): 8481.172797
  theta1 (slope):     -0.021274
Model saved to theta.csv
```

**What happened:**
1. Loaded 24 car records from data.csv
2. Ran gradient descent 1000 times
3. Found the best theta0 and theta1
4. Saved them to theta.csv

### Step 2: Predict Prices

```bash
python predict.py
```

**Example session:**
```
Enter the mileage (km): 100000
Estimated price for 100000 km: 6353.80
```

**Try different values:**
```
Enter the mileage (km): 50000
Estimated price for 50000 km: 7417.50

Enter the mileage (km): 200000
Estimated price for 200000 km: 4226.41
```

### Step 3: Visualize the Results (Bonus)

```bash
python visualize.py
```

**What you'll see:**
- Blue dots: actual data points from the dataset
- Red line: the regression line your model learned

The line should pass through the middle of the dots, showing a good fit.

### Step 4: Understand Convergence with plot_convergence.py

```bash
python plot_convergence.py
```

This tool helps you **visualize** how different learning rates affect training. It's essential for understanding gradient descent and choosing good hyperparameters.

#### What the Tool Does

1. Loads the dataset (same as train.py)
2. Runs gradient descent with different learning rates (0.1, 0.5, 1.0, 2.0)
3. Records the **cost** (MSE) at each iteration
4. Plots cost vs iterations for each learning rate

#### The Graphs Explained

**Graph 1 (Left) - Comparing Learning Rates:**

```
Cost
  ^
  |*                          <- All start at same cost (high error)
  | *
  |  \  *                     <- lr=0.1 (slow descent)
  |   \   *
  |    \    *---*---*         <- lr=0.5 (medium speed)
  |     \
  |      \---*---*---*        <- lr=1.0 (fast, reaches bottom quickly)
  |
  |  *   *   *   *   *        <- lr=2.0 (bouncing, never settles)
  +-------------------------> Iterations
```

**What each line means:**

| Line Color | Learning Rate | Behavior |
|------------|---------------|----------|
| Blue | 0.1 | Slow, steady decrease. Safe but needs many iterations |
| Orange | 0.5 | Medium speed. Good balance |
| Green | 1.0 | Fast decrease. Reaches minimum quickly |
| Red | 2.0 | Stays high or oscillates. **DIVERGES** |

**Graph 2 (Right) - Same comparison:**

Shows the same learning rates to confirm the pattern.

#### How to Read the Graphs

**Good convergence looks like:**
```
Cost
  ^
  |*
  | \
  |  \
  |   \
  |    \____________________  <- Flat line = converged
  +-------------------------> Iterations
```

**Bad (divergence) looks like:**
```
Cost
  ^
  |      *       *       *   <- Bouncing up and down
  |    /   \   /   \   /
  |  /       *       *       <- Never settles
  |*
  +-------------------------> Iterations
```

#### What to Look For

1. **Convergence point**: Where the line flattens out
   - This tells you how many iterations you need

2. **Speed of descent**: How quickly the line drops
   - Faster = higher learning rate (good, if stable)

3. **Stability**: Does the line bounce or stay smooth?
   - Bouncing = learning rate too high

4. **Final cost**: Where the line ends up
   - Lower = better fit

#### Using plot_convergence.py to Choose Hyperparameters

**Step 1:** Run the tool and look at the graphs

**Step 2:** Identify which learning rates converge
- In our case: 0.1, 0.5, 1.0 all converge
- 2.0 diverges

**Step 3:** Pick the fastest one that's stable
- lr = 1.0 converges quickly and smoothly
- lr = 1.5 or 1.7 might be even faster (you can modify the code to test)

**Step 4:** Note where the cost flattens
- This is your minimum iterations needed
- Add some buffer (e.g., if it flattens at 150, use 200)

#### Example Analysis

Looking at our dataset:

```
lr = 0.1: Cost flattens around iteration 1500
lr = 0.5: Cost flattens around iteration 300
lr = 1.0: Cost flattens around iteration 150
lr = 2.0: Cost never flattens (diverges)
```

**Conclusion:** Use lr = 1.0 with 200 iterations (or lr = 0.5 with 400 iterations for safety).

---

## Summary

### Key Concepts

| Concept | Definition | Example |
|---------|------------|---------|
| **θ₀ (theta0)** | Y-intercept, base price | 8481 |
| **θ₁ (theta1)** | Slope, price change per km | -0.021 |
| **Cost (MSE)** | Measure of prediction error | Lower = better |
| **Learning Rate** | Step size (0 < lr < 2) | 0.1 (safe), 1.0 (fast) |
| **Iterations** | Number of gradient descent steps | Until converged |
| **Convergence** | When values stop changing | Changes < 0.0001 |
| **Normalization** | Scaling data to [0,1] | Required for stability |

### The Algorithm in Plain English

1. **Start** with theta0 = 0 and theta1 = 0 (a flat line at y = 0)
2. **Predict** prices for all cars using current thetas
3. **Calculate** how wrong we are (the cost)
4. **Adjust** thetas to reduce the error (gradient descent)
5. **Repeat** steps 2-4 until the adjustments become tiny
6. **Done!** We have the best theta0 and theta1

### The Final Model

```
price = 8481 + (-0.021 × mileage)
```

This means:
- A brand new car (0 km) would cost 8481
- Each kilometer driven reduces the price by 0.021
- A car with 100,000 km costs about 6353
