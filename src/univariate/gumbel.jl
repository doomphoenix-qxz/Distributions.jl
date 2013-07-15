immutable Gumbel <: ContinuousUnivariateDistribution
    mu::Float64   # location
    beta::Float64 # scale
    function Gumbel(mu::Real, beta::Real)
        if beta <= 0
            error("beta must be positive")
        end
        new(float64(mu), float64(beta))
    end
end

Gumbel() = Gumbel(0.0, 1.0)

const DoubleExponential = Gumbel

cdf(d::Gumbel, x::Real) = exp(-exp((d.mu - x) / d.beta))

logcdf(d::Gumbel, x::Real) = -exp((d.mu - x) / d.beta)

entropy(d::Gumbel) = log(d.beta) - digamma(1.0) + 1.0

insupport(::Gumbel, x::Real) = isfinite(x)
insupport(::Type{Gumbel}, x::Real) = isfinite(x)

kurtosis(d::Gumbel) = 2.4

mean(d::Gumbel) = d.mu - d.beta * digamma(1.0)

median(d::Gumbel) = d.mu - d.beta * log(log(2.0))

modes(d::Gumbel) = [d.mu]

function pdf(d::Gumbel, x::Real)
    z = (x - d.mu) / d.beta
    exp(-z - exp(-z)) / d.beta
end

function logpdf(d::Gumbel, x::Real)
    z = (x - d.mu) / d.beta
    -z - exp(-z) - log(d.beta)
end

quantile(d::Gumbel, p::Real) = d.mu - d.beta * log(-log(p))

rand(d::Gumbel) = d.mu - d.beta * log(-log(rand()))

skewness(d::Gumbel) = 12.0 * sqrt(6.0) * zeta(3.0) / pi^3

var(d::Gumbel) = pi^2 / 6.0 * d.beta^2
