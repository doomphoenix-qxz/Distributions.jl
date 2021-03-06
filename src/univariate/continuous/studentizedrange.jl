using SpecialFunctions: gamma, erf
using ForwardDiff: derivative
using QuadGK: quadgk
"""
    StudentizedRange(k,ν)

The *Studentized range distribution* has probability density function

```math
f(q; k, \\nu) = \\frac{\\sqrt{2\\pi}k(k-1)\\nu^{\\nu/2}}{\\Gamma(\nu/2)2^{\\nu/2 - 1}}
\\int_{0}^{\\infty}\\varphi(\\sqrt{\\nu}x)\\Big[\\int_{-\\infty}^{\\infty}\\varphi(u)
\\varphi(u-qx)(\\Phi(u)\\Phi(u-qx)^{k-2})du\\Big]dx, \\quad x > 0, \\quad k > 1
```


```julia
StudentizedRange(n, ν)    # Studentized range distribution with number of samples n and degrees of freedom ν

params(d)          # Get the parameters, i.e. (k, ν)
dof(d)             # Get the degrees of freedom, i.e. ν
```

External links

* [Studentized range distribution on Wikipedia](https://en.wikipedia.org/wiki/Studentized_range_distribution)

"""
struct StudentizedRange{T<:Real} <: ContinuousUnivariateDistribution
    k::T
    ν::T
    coeff_pdf::T
    coeff_cdf::T
end

function StudentizedRange{T}(k, ν) where T
    (k, ν) = (Float64(k), Float64(ν))
    coeff_pdf = (√(2π) * k * (k-1) * ν^(ν/2)) / (gamma(ν/2) * 2^(ν/2 - 1))
    coeff_cdf = (k * ν^(ν/2)) / (gamma(ν/2) * 2^(ν/2 - 1))
    return StudentizedRange(k, ν, coeff_pdf, coeff_cdf)
end

StudentizedRange(k::T, ν::T) where {T<:Real} = StudentizedRange{T}(k, ν)
StudentizedRange(k::Real, ν::Real) = StudentizedRange(promote(k, ν)...)
StudentizedRange(k::Integer, ν::Integer) = StudentizedRange(Float64(k), Float64(ν))
StudentizedRange(a) = StudentizedRange(a,a)
StudentizedRange() = StudentizedRange(2,2)

@distr_support StudentizedRange 0.0 Inf

### Conversions
function convert(::Type{StudentizedRange{T}}, k::Real, ν::Real) where T<:Real
    StudentizedRange(T(k), T(ν))
end
function convert(::Type{StudentizedRange{T}}, d::StudentizedRange{S}) where {T <: Real, S <: Real}
    StudentizedRange(T(d.k), T(d.ν))
end

### Parameters

params(d::StudentizedRange) = (d.k, d.ν)
dof(d::StudentizedRange) = d.ν

### Statistics

mean(d::StudentizedRange{T}) = T(NaN)
mode(d::StudentizedRange{T}) = T(NaN)
var(d::StudentizedRange{T}) = T(NaN)
skewness(d::StudentizedRange{T}) = T(NaN)
entropy(d::StudentizedRange{T}) = T(NaN)

### Evaluation

# Helper functions for cdf and pdf of standard normal.
function 𝚽(x)
    return (1+erf(x / √2)) / 2
end

function ϕ(x)
    return derivative(𝚽, x)
end


function cdf(d::StudentizedRange, q)
    function outer(x)

        function inner(u)
            return ϕ(u) * (𝚽(u) - 𝚽(u - q*x))^(d.k-1)
        end
        inner_part = quadgk(inner, -Inf, Inf)[1]
        return inner_part * x^(d.ν-1) * exp(-x^2*d.ν / 2)
    end
    integral = quadgk(outer, 0.0, Inf)[1]
    return integral * d.coeff_cdf
end

function pdf(d::StudentizedRange, q)

    function outer(x)

        function inner(u)
            return ϕ(u) * ϕ(u - q*x) * (𝚽(u) - 𝚽(u - q*x))^(d.k-2)
        end
        inner_part = quadgk(inner, -Inf, Inf)[1]
        return inner_part * x^d.ν * ϕ(x*√(d.ν))
    end
    integral = quadgk(outer, 0.0, Inf)[1]
    return integral * d.coeff_pdf
end

logpdf(d::StudentizedRange, q) = log(pdf(d, q))

# To get quantile to work correctly I had to implement my quick naive version of
# the bisection method. I'm not sure why, but trying to do this with Roots.jl
# was WAY too slow. Like 30+ seconds.
function simple_bisection(f::Function, brackets, abstol=10.0^-6, maxeval=1e3)
    if brackets[1] > brackets[2]
        xmax, xmin = brackets
    else
        xmin, xmax = brackets
    end
    @assert f(xmin) * f(xmax) < 0

    a = xmin
    b = xmax
    error = 1
    numeval = 0
    while error > abstol

        numeval += 1
        if numeval > maxeval break end

        fnew = f((a+b)/2)
        error = abs(fnew)
        if fnew * f(a) < 0
            b = (a+b)/2
        elseif fnew * f(b) < 0
            a = (a+b)/2
        else
            throw(BoundsError("Algorithm failed to converge. Sorry :("))
        end

    end
    return (a+b)/2
end

function quantile(d::StudentizedRange, x)
    @assert 0.0 <= x < 1.0
    if x == 0.0 return 0.0 end

    return simple_bisection(y -> cdf(d, y) - x, [0.0, 1000.0])
end

median(d::StudentizedRange) = quantile(d, 0.5)
