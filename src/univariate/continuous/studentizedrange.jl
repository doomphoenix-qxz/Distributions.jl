using SpecialFunctions: gamma, erf
using ForwardDiff: derivative
using QuadGK: quadgk

struct StudentizedRange{T<:Integer}
    k::T
    ν::T
    coeff::Float64
end

function StudentizedRange(k, ν)
    (k, ν) = (Int64(k), Int64(ν))
    coeff = (√(2π) * k * (k-1) * ν^(ν/2)) / (gamma(ν/2) * 2^(ν/2 - 1))
    return StudentizedRange(k, ν, coeff)
end

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
    return integral * (d.k * d.ν^(d.ν/2)) / (gamma(d.ν/2) * 2^(d.ν/2 - 1))
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
    return integral * d.coeff
end

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
    return b
end

function quantile(d::StudentizedRange, x)
    @assert 0.0 <= x < 1.0
    if x == 0.0 return 0 end

    return simple_bisection(y -> cdf(d, y) - x, [0.0, 1000.0])
end
