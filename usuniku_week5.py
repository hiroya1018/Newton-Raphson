import mpmath as mp
import matplotlib.pyplot as plt


# 内部計算の精度（有効数字）
mp.mp.dps = 80

L = mp.mpf("1.0")

# 計算するモードとNewton–Raphson法の初期値
mode_numbers = [1, 2, 3, 17]
initial_guesses = ["1.9", "4.7", "7.9", "51.8"]

# 小数点以下20桁で比較するための倍率
decimal_places = 20
scale = mp.mpf(10) ** decimal_places


def f(lam):
    """固有値方程式 f(λ) = 0"""
    z = lam * L
    return 1 + mp.cos(z) * mp.cosh(z)


def df(lam):
    """固有値方程式の導関数 f'(λ)"""
    z = lam * L
    return L * (
        mp.cos(z) * mp.sinh(z)
        - mp.sin(z) * mp.cosh(z)
    )


def rounded_to_20_decimal_places(value):
    """数値を小数点以下20桁に丸めた整数を返す"""
    return mp.nint(value * scale)


def newton_raphson_20_digits(initial_guess, max_iterations=100):
    """小数点以下20桁まで値が変わらなくなるまでNewton法を反復"""
    lam = mp.mpf(initial_guess)

    for iteration in range(1, max_iterations + 1):
        derivative = df(lam)

        if abs(derivative) < mp.mpf("1e-70"):
            raise ZeroDivisionError(
                "導関数が小さすぎます。初期値を確認してください。"
            )

        next_lam = lam - f(lam) / derivative

        # 連続する値が小数点以下20桁まで一致したら収束とする
        if rounded_to_20_decimal_places(lam) == rounded_to_20_decimal_places(next_lam):
            return next_lam, iteration

        lam = next_lam

    raise RuntimeError(
        f"初期値 {initial_guess} からの計算が収束しませんでした。"
    )


def mode_shape(lam, sigma, x):
    """A=1に正規化したモード形状"""
    z = lam * x

    # y(x) = A cos(λx) + B sin(λx)
    #      + C cosh(λx) + D sinh(λx)
    return (
        mp.cos(z)
        - sigma * mp.sin(z)
        - mp.cosh(z)
        + sigma * mp.sinh(z)
    )


# 固有値を計算
roots = []

print("固有値の計算結果")
for n, guess in zip(mode_numbers, initial_guesses):
    root, iterations = newton_raphson_20_digits(guess)
    roots.append(root)

    print(
        f"Mode {n:2d}: λ = {mp.nstr(root, 40)}"
        f"  (反復回数: {iterations})"
    )


# 係数を表示
print("\nモード形状の係数（A=1に正規化）")
print("y(x) = A cos(λx) + B sin(λx) + C cosh(λx) + D sinh(λx)")

for n, lam in zip(mode_numbers, roots):
    z = lam * L

    sigma = (mp.cosh(z) + mp.cos(z)) / (
        mp.sinh(z) + mp.sin(z)
    )

    A = mp.mpf("1")
    B = -sigma
    C = mp.mpf("-1")
    D = sigma

    print(f"\nMode {n}")
    print(f"  A = {mp.nstr(A, 30)}")
    print(f"  B = {mp.nstr(B, 30)}")
    print(f"  C = {mp.nstr(C, 30)}")
    print(f"  D = {mp.nstr(D, 30)}")


# モード形状をプロット
number_of_points = 1200
x_values = [L * i / number_of_points for i in range(number_of_points + 1)]

for n, lam in zip(mode_numbers, roots):
    z = lam * L

    sigma = (mp.cosh(z) + mp.cos(z)) / (
        mp.sinh(z) + mp.sin(z)
    )

    # 任意精度で形状を計算してから、グラフ用に正規化
    y_values = [
        mode_shape(lam, sigma, x)
        for x in x_values
    ]

    max_amplitude = max(abs(y) for y in y_values)
    normalized_y = [y / max_amplitude for y in y_values]

    plt.plot(
        [float(x) for x in x_values],
        [float(y) for y in normalized_y],
        label=f"Mode {n}"
    )

plt.xlabel("x")
plt.ylabel("Normalized mode shape")
plt.title("Cantilever beam vibration modes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()