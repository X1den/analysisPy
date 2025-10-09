import numpy as np
import math

a, b = 1.5, 3.75
E0 = 1e-3

def f(x):
    return x**3 * np.exp(-0.12 * x**1.5)

true_value = np.trapz(f(np.linspace(a, b, 1000000)),
                      np.linspace(a, b, 1000000))

def midpoint_rule(f, a, b, n, result = 0):
    h = (b - a) / n
    for i in range(n): 
        x = a + h * (i + 0.5)
        result += f(x)

    return result * h

def trapezoidal_rule(f, a, b, n, result = 0):
    h = (b - a) / n
    for i in range(n):
        x1 = a + i * h 
        x2 = a + (i + 1) * h
        result += 0.5 * (x2 - x1) * (f(x1) + f(x2))

    return result

def simpson_rule(f, a, b, n, result = 0):
    if n <= 0: return 0.0
    if n % 2 != 0: n += 1

    h = (b - a) / n
    for i in range(n):
        x1 = a + i * h
        x2 = a + (i + 1) * h
        result += (x2 - x1) / 6.0 * (f(x1) + 4.0 * f(0.5 * (x1 + x2)) + f(x2))

    return result

def find_n(method, name):
    n = 2
    while True:
        I = method(f, a, b, n)
        if abs(I - true_value) <= E0:
            return n, I
        n += 1

n_mid, I_mid = find_n(midpoint_rule, "прямоугольников")
n_trap, I_trap = find_n(trapezoidal_rule, "трапеций")
n_simp, I_simp = find_n(simpson_rule, "Симпсона")

print(f"Истинное значение I ≈ {true_value:.6f}\n")
print("Результаты:")
print(f"  Прямоугольники: n = {n_mid:3d}, I = {I_mid:.6f}, ошибка = {abs(I_mid - true_value):.6e}")
print(f"  Трапеции:       n = {n_trap:3d}, I = {I_trap:.6f}, ошибка = {abs(I_trap - true_value):.6e}")
print(f"  Симпсон:        n = {n_simp:3d}, I = {I_simp:.6f}, ошибка = {abs(I_simp - true_value):.6e}")