"""
    Beta(α,β)

The *Beta distribution* has probability density function

```math
f(x; \\alpha, \\beta) = \\frac{1}{B(\\alpha, \\beta)}
 x^{\\alpha - 1} (1 - x)^{\\beta - 1}, \\quad x \\in [0, 1]
```

The Beta distribution is related to the [`Gamma`](@ref) distribution via the
property that if ``X \\sim \\operatorname{Gamma}(\\alpha)`` and ``Y \\sim \\operatorname{Gamma}(\\beta)``
independently, then ``X / (X + Y) \\sim Beta(\\alpha, \\beta)``.


```julia
Beta()        # equivalent to Beta(1, 1)
Beta(a)       # equivalent to Beta(a, a)
Beta(a, b)    # Beta distribution with shape parameters a and b

params(d)     # Get the parameters, i.e. (a, b)
```

External links

* [Beta distribution on Wikipedia](http://en.wikipedia.org/wiki/Beta_distribution)

"""
struct Beta{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T

    function Beta{T}(α::T, β::T) where T
        @check_args(Beta, α > zero(α) && β > zero(β))
        new{T}(α, β)
    end
end

Beta(α::T, β::T) where {T<:Real} = Beta{T}(α, β)
Beta(α::Real, β::Real) = Beta(promote(α, β)...)
Beta(α::Integer, β::Integer) = Beta(Float64(α), Float64(β))
Beta(α::Real) = Beta(α, α)
Beta() = Beta(1, 1)

@distr_support Beta 0.0 1.0

#### Conversions
function convert(::Type{Beta{T}}, α::Real, β::Real) where T<:Real
    Beta(T(α), T(β))
end
function convert(::Type{Beta{T}}, d::Beta{S}) where {T <: Real, S <: Real}
    Beta(T(d.α), T(d.β))
end

#### Parameters

params(d::Beta) = (d.α, d.β)
@inline partype(d::Beta{T}) where {T<:Real} = T


#### Statistics

mean(d::Beta) = ((α, β) = params(d); α / (α + β))

function mode(d::Beta)
    (α, β) = params(d)
    (α > 1 && β > 1) || error("mode is defined only when α > 1 and β > 1.")
    return (α - 1) / (α + β - 2)
end

modes(d::Beta) = [mode(d)]

function var(d::Beta)
    (α, β) = params(d)
    s = α + β
    return (α * β) / (abs2(s) * (s + 1))
end

meanlogx(d::Beta) = ((α, β) = params(d); digamma(α) - digamma(α + β))

varlogx(d::Beta) = ((α, β) = params(d); trigamma(α) - trigamma(α + β))
stdlogx(d::Beta) = sqrt(varlogx(d))

function skewness(d::Beta)
    (α, β) = params(d)
    if α == β
        return zero(α)
    else
        s = α + β
        (2(β - α) * sqrt(s + 1)) / ((s + 2) * sqrt(α * β))
    end
end

function kurtosis(d::Beta)
    α, β = params(d)
    s = α + β
    p = α * β
    6(abs2(α - β) * (s + 1) - p * (s + 2)) / (p * (s + 2) * (s + 3))
end

function entropy(d::Beta)
    α, β = params(d)
    s = α + β
    lbeta(α, β) - (α - 1) * digamma(α) - (β - 1) * digamma(β) +
        (s - 2) * digamma(s)
end


#### Evaluation

@_delegate_statsfuns Beta beta α β

gradlogpdf(d::Beta{T}, x::Real) where {T<:Real} =
    ((α, β) = params(d); 0 <= x <= 1 ? (α - 1) / x - (β - 1) / (1 - x) : zero(T))


#### Sampling

rand(d::Beta) = StatsFuns.RFunctions.betarand(d.α, d.β)


#### Fit model

# TODO: add MLE method (should be similar to Dirichlet)

# This is a moment-matching method (not MLE)
#
function fit(::Type{Beta}, x::AbstractArray{T}) where T<:Real
    x_bar = mean(x)
    v_bar = varm(x, x_bar)
    α = x_bar * (((x_bar * (1 - x_bar)) / v_bar) - 1)
    β = (1 - x_bar) * (((x_bar * (1 - x_bar)) / v_bar) - 1)
    Beta(α, β)
end
