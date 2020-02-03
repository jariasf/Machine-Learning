'''
-- Problem: Polynomial Regression: Office Prices
-- Author: Jhosimar George Arias Figueroa
'''

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

def normalized_distance(computed, expected):
   return abs(computed-expected)/expected  

# Read input
# Train
F, N = map(int, input().split())
data = np.array([input().split() for _ in range(N)], float)
X = data[:,:-1]
y = data[:, -1]
# Test
T = int(input())
X_test = np.array([ input().split() for _ in range(T)], float)

# Polynomial features
poly = PolynomialFeatures(3)
X_poly = poly.fit_transform(X)

# Model
model = LinearRegression().fit(X_poly, y)
X_poly_test = poly.fit_transform(X_test)
y_pred = model.predict(X_poly_test)
print(*y_pred, sep='\n' )
