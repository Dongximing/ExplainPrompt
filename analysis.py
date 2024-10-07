import numpy as np
from scipy.stats import spearmanr

# Create a 2D array with sample data
# Here, we assume each row is a variable and each column is an observation
data = np.array([
    [1, 2, 3, 4, 5],     # Variable 1
    [2, 3, 2, 1, 0],     # Variable 2
    [3.5, 2.5, 0.5, 1.5, 3.0]  # Variable 3
]).T  # Transpose to make columns into variables

# Calculate the Spearman correlation coefficient and p-value
correlation, p_value = spearmanr(data)

print("Spearman correlation coefficient matrix:\n", correlation)
print("P-value matrix:\n", p_value)
