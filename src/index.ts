// noinspection JSUnusedGlobalSymbols,JSUnusedLocalSymbols

import {Decimal as BN} from "decimal.js";

BN.set({precision: 100});

const EPSILON = new BN(1).div("10000000");
const EPSILON_2 = EPSILON.pow(2);
const EPSILON_3 = EPSILON.pow(3);
const EPSILON_MAX = EPSILON.pow(10);
const _2_EPSILON = EPSILON.times(2);
const _3_EPSILON = EPSILON.times(3);
const OVER_EPSILON = new BN(1).div(EPSILON);
const OVER_EPSILON_2 = OVER_EPSILON.pow(2);
const OVER_EPSILON_3 = OVER_EPSILON.pow(3);
const ZERO = new BN(0);

const SolveDefaultOptions = {
    x: ZERO,
    iterations: 1000
};

const SolvePolynomialDefaultOptions = {
    x: ZERO,
    iterations: 1000
};

function makePolynomial(size: number): BN[] {
    const p = [];
    for (let i = 0; i < size; i++) p[i] = ZERO;
    return p;
}

function factorial(n: number) {
    let product = 1;
    for (let i = 2; i <= n; i++) {
        product *= i;
    }
    return product;
}

function oneOverFactorial(n: number) {
    let product = 1;
    for (let i = 2; i <= n; i++) {
        product /= i;
    }
    return product;
}

function permutation(a: number, b: number) {
    if (a < b) throw new Error("Invalid permutation argument.");
    if (b === 0) return 1;
    if (a === b) return factorial(a);
    let product = 1;
    // 6, 2 -> 5 * 6
    for (let i = a - b + 1; i <= a; i++) {
        product *= i;
    }
    return product;
}

function combination(a: number, b: number) {
    if (a < b) throw new Error("Invalid combination argument.");
    if (a === b || b === 0) return 1;
    if (b > a / 2) b = a - b;
    if (b === 1) return a;
    return permutation(a, b) / factorial(b);
}

function addPolynomials(...polynomials: BN[][]) {
    let max_size = 0;
    for (let i = 0; i < polynomials.length; i++) {
        const p = polynomials[i];
        if (p.length > max_size) max_size = p.length;
    }
    const poly = makePolynomial(max_size);
    for (let i = 0; i < polynomials.length; i++) {
        const p = polynomials[i];
        for (let j = 0; j < p.length; j++) {
            poly[j] = poly[j].plus(p[j]);
        }
    }
    return poly;
}

function multiplyTwoPolynomials(a: BN[], b: BN[]) {
    let max_size = (a.length - 1) + (b.length - 1) + 1;
    const poly = makePolynomial(max_size);
    for (let i = 0; i < a.length; i++) {
        for (let j = 0; j < b.length; j++) {
            poly[i + j] = poly[i + j].plus(a[i].times(b[j]));
        }
    }
    return poly;
}

function multiplyPolynomials(...p: BN[][]) {
    if (p.length === 0) return [ZERO];
    let poly = p[0];
    for (let i = 1; i < p.length; i++) {
        poly = multiplyTwoPolynomials(poly, p[i]);
    }
    return poly;
}

function polynomialPowerToInt(poly: BN[], power: number) {
    if (power === 0) return makePolynomial(1);
    if (power === 1) return [...poly];
    const p = poly;
    for (let i = 1; i < power; i++) {
        poly = multiplyTwoPolynomials(poly, p);
    }
    return poly;
}

function solveFunction(f: (bn: BN) => BN, _options?: Partial<typeof SolveDefaultOptions>) {
    const options = {...SolveDefaultOptions, ...(typeof _options === "object" ? _options : {})};
    let x = options.x;
    // Newton's method
    // x_{n+1} = x_n - f(x_n) / f'(x_n)
    // x_{n+1} = x_n - f(x_n) * ε / (f(x_n + ε) - f(x_n))
    for (let i = 0; i < options.iterations; i++) {
        const fx = f(x);
        if (fx.abs().lessThan(EPSILON_MAX)) return x;
        x = x.minus(fx.times(EPSILON).dividedBy(f(x.plus(EPSILON)).minus(fx)));
    }
    return x;
}

function evaluatePolynomial(polynomial: BN[], x: BN) {
    let res = ZERO;
    for (let n = 0; n < polynomial.length; n++) {
        res = res.plus(polynomial[n].times(x.pow(n)));
    }
    return res;
}

function derivePolynomial(polynomial: BN[]) {
    // basic power rule
    // [1, 2, 3, 4] 4x^3 + 3x^2 + 2x + 1
    // [2 * 1, 3 * 2, 4 * 3]
    const result = makePolynomial(polynomial.length - 1);
    for (let n = 0; n < polynomial.length - 1; n++) {
        result[n] = polynomial[n + 1].times(n + 1);
    }
    return result;
}

function approximate1stDerivativeFunction(f: (bn: BN) => BN, at: BN) {
    // f(x + h) - f(x)
    return f(at.plus(EPSILON)).minus(f(at)).times(OVER_EPSILON);
}

function approximate2ndDerivativeFunction(f: (bn: BN) => BN, at: BN) {
    // f(x + 2h) - 2f(x + h) + f(x)
    return (f(at.plus(_2_EPSILON)).minus(f(at.plus(EPSILON)).times(2)).plus(f(at))).times(OVER_EPSILON_2);
}

function approximate3rdDerivativeFunction(f: (bn: BN) => BN, at: BN) {
    // f(x + 3h) - 3f(x + 2h) + 3f(x + h) - f(x)
    return (f(at.plus(_3_EPSILON)).minus(f(at.plus(_2_EPSILON)).times(3)).plus(f(at.plus(EPSILON)).times(3)).minus(f(at))).times(OVER_EPSILON_2);
}

function approximateNthDerivativeFunction(f: (bn: BN) => BN, n: number, x: BN) {
    if (n === 0) return f(x);
    if (n === 1) return approximate1stDerivativeFunction(f, x);
    if (n === 2) return approximate2ndDerivativeFunction(f, x);
    if (n === 3) return approximate3rdDerivativeFunction(f, x);
    // changing the formula I found so that it is suitable for computing:
    // f^(n)(x) = sum_{i=0}^n (1/ε)^n (-1)^i C(n, i) f(x + (n - i) ε)
    // f^(n)(x) = sum_{i=0}^n (1/ε)^n (-1)^(n + i) C(n, i) f(x + iε)

    // f^(n)(x) = sum_{i=0}^n (-1/ε)^n (-1)^i C(n, i) f(x + iε)

    let sum = ZERO;
    const constant = OVER_EPSILON.pow(n).times((n % 2) * 2 - 1);
    for (let i = 0; i <= n; i++) {
        // (-1)^i -> (i % 2) * 2 - 1
        sum = sum.plus(f(x.plus(EPSILON.times(i))).times(((i % 2) * 2 - 1) * combination(n, i)));
    }
    return sum.times(constant);
}

function integratePolynomial(polynomial: BN[]) {
    // basic power rule
    // [1, 2, 3, 4] 4x^3 + 3x^2 + 2x + 1
    // [0, 1 / 1, 2 / 2, 3 / 3, 4 / 4]
    const result = makePolynomial(polynomial.length + 1);
    result[0] = ZERO;
    for (let i = 1; i < polynomial.length + 1; i++) {
        result[i] = polynomial[i - 1].div(i);
    }
    return result;
}

function polynomialToFunction(polynomial: BN[]): (x: BN) => BN {
    return (x: BN) => evaluatePolynomial(polynomial, x);
}

function functionToPolynomial(f: (bn: BN) => BN, degree: number) {
    // taylor series
    // f(x) = sum_{n=0}^∞ (f^(n)(a) (x-a)^n) / n!
    // maclaurin series
    // f(x) = sum_{n=0}^∞ (f^(n)(0) x^n) / n!
    const polynomial = makePolynomial(degree + 1);
    for (let n = 0; n < degree + 1; n++) {
        polynomial[n] = approximateNthDerivativeFunction(f, n, ZERO).times(oneOverFactorial(n));
    }
    return polynomial;
}

function solvePolynomial(polynomial: BN[], _options?: Partial<typeof SolvePolynomialDefaultOptions>) {
    const options = {...SolveDefaultOptions, ...(typeof _options === "object" ? _options : {})};
    const derivative = derivePolynomial(polynomial);
    let x = options.x;
    // x_{n+1} = x_n - f(x_n) / f'(x_n)
    for (let i = 0; i < options.iterations; i++) {
        const fx = evaluatePolynomial(polynomial, x);
        if (fx.abs().lessThan(EPSILON_MAX)) return x;
        const dfx = evaluatePolynomial(derivative, x);
        x = x.minus(fx.div(dfx.isZero() ? EPSILON_3 : dfx));
    }
    return x;
}

function stringToPolynomial(text: string) {
    text = text.replaceAll(/[ ^]/g, "");
    let maxDegree = 0;
    const a = [...text.matchAll(/[+-]?([\d.]*x\d+|[\d.]*x|[\d.]+)/g)].map(i => {
        const spl = i[0].replaceAll("+", "").split("x");
        const c = spl[0] ? parseFloat(spl[0]) : 1;
        const p = spl[1] ? parseFloat(spl[1]) : (i[0].includes("x") ? 1 : 0);
        if (maxDegree < p) maxDegree = p;
        return [c, p];
    });
    const poly = makePolynomial(maxDegree + 1);
    for (const b of a) {
        poly[b[1]] = poly[b[1]].plus(b[0]);
    }
    return poly;
}

function $poly(text: string | TemplateStringsArray) { // Usage: $poly`x^2 + 5`
    return stringToPolynomial(typeof text === "string" ? text : text[0]);
}

function polynomialToString(polynomial: BN[]) {
    let str = "";
    for (let i = polynomial.length - 1; i >= 0; i--) {
        let v = polynomial[i];
        if (v.isZero()) continue;
        if (!str) {
            if (!v.eq(1) || i === 0) str += v;
        } else str += (!v.isNegative() ? " + " : " - ") + (v.eq(1) && i !== 0 ? "" : v.abs());
        if (i !== 0) {
            str += "x";
            if (i > 1) str += "^" + i;
        }
    }
    return str;
}