#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

// f(z) = 1 + cos(z) cosh(z)
double f(double z) {
    return 1.0 + cos(z) * cosh(z);
}

// f'(z)
double df(double z) {
    return -sin(z) * cosh(z) + cos(z) * sinh(z);
}

// Newton-Raphson法で、指定した区間内の根を探す
double newton(double initial, double left, double right) {
    double z = initial;

    for (int iter = 0; iter < 100; ++iter) {
        const double deriv = df(z);
        if (abs(deriv) < 1e-14) {
            throw runtime_error("導関数が小さすぎます。初期値を確認してください。");
        }

        double next = z - f(z) / deriv;

        // 発散した場合は区間中央へ戻す
        if (!isfinite(next) || next <= left || next >= right) {
            next = (left + right) / 2.0;
        }

        if (abs(next - z) < 1e-13) {
            return next;
        }
        z = next;
    }

    throw runtime_error("Newton-Raphson法が収束しませんでした。");
}

int main() {
    const double L = 1.0;
    const vector<int> modes = { 20, 21, 22, 50 };

    cout << fixed << setprecision(12);

    for (int n : modes) {
        const double guess = (n - 0.5) * acos(-1.0);
        const double halfWidth = 0.2;
        const double left = guess - halfWidth;
        const double right = guess + halfWidth;

        const double z = newton(guess, left, right);
        const double lambda = z / L;

        // phi(0)=1 となるよう正規化
        const double A = -1.0;
        const double C = 1.0;
        const double B = (cosh(z) + cos(z)) / (sinh(z) + sin(z));
        const double D = -B;

        cout << "n = " << n << '\n'
            << "  lambda*L = " << z << '\n'
            << "  lambda   = " << lambda << '\n'
            << "  A = " << A << ", B = " << B
            << ", C = " << C << ", D = " << D << '\n'
            << "  residual = " << f(z) << "\n\n";
    }

    return 0;
}