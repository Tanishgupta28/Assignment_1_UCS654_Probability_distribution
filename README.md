# 📊 Learning Probability Density Functions using Roll-Number Parameterized Non-Linear Model

## 📌 Assignment Overview

This project learns a probability density function (PDF) from NO₂ air-quality data using a roll-number-parameterized non-linear transformation.

---

## Step 1 — Data Collection

- The NO₂ pollutant values are extracted from the dataset.
- Missing values are removed to ensure reliable statistics.
- Only valid numeric entries are used.

---

## 📂 Dataset

**India Air Quality Dataset**  
Source: Kaggle  
https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data

Feature used:


---

## 🔢 Step-2: Non-Linear Transformation

Each NO₂ value (x) is transformed into (z):

\[
z = x + a_r * sin(b_r * x)
\]

Where:

\[
a_r = 0.05 * (r % 7)
\]

\[
b_r = 0.3 * ((r % 5) + 1)
\]

`r` = University roll number

### Why this step?

- Introduces controlled non-linearity
- Makes each student’s dataset unique
- Simulates feature engineering
- Helps observe distributional changes

---

## 📈 Step-3: Statistical Modeling

We assume the transformed data follows a Gaussian-like distribution:

\[
p^​(z)=c * exp(−λ * (z−μ)^2)
\]

Parameters to learn:

- μ (mean)
- λ (precision parameter)
- c (normalization constant)

This is equivalent to a normal distribution written in exponential form.

---

## 🧮 Parameter Estimation

Using statistical estimation:

### Mean (μ)

Represents the center of the distribution.

\[
mu = mean(z)
\]

### Variance (σ²)

Measures spread of data.

\[
var = var(z)
\]

### Precision (λ)

Inverse spread measure.

\[
lambda_est = 1/(2*var)
\]

### Normalization Constant (c)

Ensures total probability equals 1.

\[
c_est = sqrt(lambda_est/pi)
\]

---

# 📊 Result Table

| Parameter | Meaning | Estimated Value |
|----------|--------|----------------|
| μ | Mean of z | 25.809622897811263 |
| Variance | Spread of z | 342.36339017375917 |
| λ | Precision | 0.001460436525489001 |
| c | Normalization constant | 0.021560876239314918 |

---

# 📈 Result Graph

The result graph compares:

1. Histogram of transformed data (z)
2. Learned probability density function

---

## Graph Code

```python
z_range = np.linspace(min(z), max(z), 500)

pdf = c_est * np.exp(-lambda_est * (z_range - mu_est)**2)

plt.figure(figsize=(8,5))
plt.hist(z, bins=40, density=True, alpha=0.5, label="Data Histogram")
plt.plot(z_range, pdf, linewidth=2, label="Learned PDF")

plt.xlabel("z values")
plt.ylabel("Density")

plt.title("Histogram vs Learned PDF")
plt.legend()
plt.show()
