# This file contains a number of additions to BlockArrays.jl. These are completely
# independent of Stheno.jl, and will (hopefully) move over to BlockArrays.jl at some point.

using FillArrays, LinearAlgebra, BlockArrays
using FillArrays: Fill

using BlockArrays: cumulsizes, blocksizes, _BlockArray, blocksizes, BlockSizes

import Base: +, *, size, getindex, eltype, copy, \, vec, getproperty, zero
import LinearAlgebra: UpperTriangular, LowerTriangular, logdet, Symmetric, transpose,
    adjoint, AdjOrTrans, AdjOrTransAbsMat, cholesky!, logdet, ldiv!, mul!, logabsdet
import BlockArrays: BlockArray, BlockVector, BlockMatrix, BlockVecOrMat, getblock,
    blocksize, setblock!, nblocks, getblock!
export unbox, BlockSymmetric

# Do some character saving.
const BV{T} = BlockVector{T}
const BM{T} = BlockMatrix{T}
const ABV{T} = AbstractBlockVector{T}
const ABM{T} = AbstractBlockMatrix{T}
const ABVM{T} = AbstractBlockVecOrMat{T}
const LUABM{T} = Union{ABM, LowerTriangular{T, <:ABM}, UpperTriangular{T}, <:ABM}

const AdjOrTransABM{T} = Union{ABM{T}, AdjOrTrans{T, <:ABM{T}}}
const AdjTransTriABM{T} = Union{AbstractTriangular{T, <:ABM{T}}, AdjOrTransABM{T}}
const TriABM{T} = Union{AbstractTriangular{T, <:ABM{T}}, ABM{T}}
const BlockTri{T} = AbstractTriangular{T, <:ABM{T}}
const BlockAdjOrTrans{T} = AdjOrTrans{T, <:ABM{T}}

unbox(X::AbstractBlockMatrix) = X
unbox(X::Symmetric) = unbox(X.data)
unbox(X::AbstractMatrix) = X



####################################### Various util #######################################

import BlockArrays: BlockVector, BlockMatrix, blocksizes, cumulsizes
export BlockVector, BlockMatrix, blocksizes, blocklengths

"""
    BlockVector(xs::AbstractVector{<:AbstractVector})

Construct a `BlockVector` from a collection of `AbstractVector`s.
"""
BlockVector(xs::AV{<:AV}) = _BlockArray(xs, convert(Vector{Int}, length.(xs)))
@adjoint function BlockVector(xs::AV{<:AV})
    x = BlockVector(xs)
    function back(Δ::BlockVector)
        @assert cumulsizes(x) == cumulsizes(Δ)
        return (Δ.blocks,)
    end
    function back(Δ::AbstractVector)
        sz = cumulsizes(x)[1]
        backs = [Δ[sz[n]:sz[n+1]-1] for n in 1:length(sz)-1]
        return (backs,)
    end
    return x, back
end
@adjoint Vector(x::BlockVector) = Vector(x), Δ->(Δ,)

"""
    BlockMatrix(Xs::Matrix{<:AbstractVecOrMat{T}}) where T

Construct a `BlockMatrix` from a matrix of `AbstractVecOrMat`s.
"""
function BlockMatrix(Xs::Matrix{<:AbstractVecOrMat{T}}) where T

    # Check that sizes make sense.
    heights, widths = size.(Xs[:, 1], 1), size.(Xs[1, :], 2)
    for q in 1:size(Xs, 2), p in 1:size(Xs, 1)
        @assert size(Xs[p, q]) == (heights[p], widths[q])
    end

    # Construct BlockMatrix.
    return _BlockArray(Xs, heights, widths)
end
@adjoint function BlockMatrix(Xs::Matrix{<:AbstractVecOrMat{T}}) where T
    X = BlockMatrix(Xs)
    function back(Δ::BlockMatrix)
        @assert cumulsizes(X) == cumulsizes(Δ)
        return (Δ.blocks,)
    end
    function back(Δ_::AbstractMatrix)
        @assert size(X) == size(Δ_)
        Δ = PseudoBlockArray(Δ_, blocksizes(X, 1), blocksizes(X, 2))
        out = ([getblock(Δ, p, q) for p in 1:nblocks(X, 1), q in 1:nblocks(X, 2)],)
        return out
    end
    return X, back
end
@adjoint Matrix(X::BlockMatrix) = Matrix(X), Δ->(Δ,)

BlockMatrix(Xs::Vector{<:AbstractVecOrMat}) = BlockMatrix(reshape(Xs, length(Xs), 1))
BlockMatrix(x::AbstractVecOrMat) = BlockMatrix([x])

"""
    BlockMatrix(xs::Vector{<:AM}, P::Int, Q::Int)

Construct a block matrix with `P` rows and `Q` columns of blocks.
"""
BlockMatrix(xs::Vector{<:AM}, P::Int, Q::Int) = BlockMatrix(reshape(xs, P, Q))

"""
    blocksizes(X::AbstractArray, d::Int)

Get a vector containing the block sizes over the `d`th dimension of `X`. 
"""
blocksizes(X::AbstractArray, d::Int) = diff(cumulsizes(X, d))
@nograd blocksizes

zero(x::AbstractBlockVector) = BlockVector([zero(getblock(x, n)) for n in 1:nblocks(x, 1)])
function zero(X::AbstractBlockMatrix)
    blocks = [zero(getblock(X, p, q)) for p in 1:nblocks(X, 1), q in 1:nblocks(X, 2)]
    return BlockMatrix(blocks)
end

#################################### BlockDiagonal #########################################


const BlockDiagonal{T, TM} = BlockMatrix{T, <:Diagonal{TM}} where {TM <: AbstractMatrix{T}}

function block_diagonal(vs::AbstractVector{<:AbstractMatrix{T}}) where {T}
    return _BlockArray(Diagonal(vs), size.(vs, 1), size.(vs, 2))
end

function LinearAlgebra.diagzero(D::Diagonal{<:AbstractMatrix{T}}, r, c) where {T}
    return Zeros{T}(size(D.diag[r], 1), size(D.diag[c], 2))
end

# Strip unhelpful wrappers to ensure that ldiv! is efficient.
strip_block(X::UpperTriangular) = X
strip_block(X::UpperTriangular{T, <:Diagonal{T}} where {T}) = X.data

function cholesky(A::BlockDiagonal)
    Cs = [strip_block(cholesky(A).U) for A in diag(A.blocks)]
    return Cholesky(BlockArrays._BlockArray(Diagonal(Cs), A.block_sizes), :U, 0)
end
function cholesky(A::Symmetric{T, <:BlockDiagonal{T}} where {T})
    Cs = [strip_block(cholesky(Symmetric(A)).U) for A in diag(A.data.blocks)]
    return Cholesky(BlockArrays._BlockArray(Diagonal(Cs), A.data.block_sizes), :U, 0)
end

function logdet(C::Cholesky{T, <:BlockDiagonal{T}} where {T})
    return 2 * sum(n->logabsdet(C.factors[Block(n, n)])[1], 1:nblocks(C.factors, 1))
end

# Because Base is dumb and hasn't implemented `logabsdet` for `Diagonal` matrices.
logabsdet(d::Diagonal) = logabsdet(UpperTriangular(d))

function ldiv!(U::UpperTriangular{T, <:BlockDiagonal{T}} where {T}, x::ABV)
    @assert are_conformal(U.data, x)
    blocks = U.data.blocks.diag
    for n in 1:nblocks(x, 1)
        setblock!(x, ldiv!(blocks[n], x[Block(n)]), n)
    end
    return x
end
\(U::UpperTriangular{T, <:BlockDiagonal{T}} where {T}, x::ABV) = ldiv!(U, copy(x))

function ldiv!(U::UpperTriangular{T, <:BlockDiagonal{T}} where {T}, X::ABM)
    @assert are_conformal(U.data, X)
    blocks = U.data.blocks.diag
    for r in 1:nblocks(X, 1)
        for c in 1:nblocks(X, 2)
            setblock!(X, ldiv!(blocks[r], X[Block(r, c)]), r, c)
        end
    end
    return X
end
\(U::UpperTriangular{T, <:BlockDiagonal{T}} where {T}, X::ABM) = ldiv!(U, copy(X))

# THIS IS ALL WRONG AND MY TESTING IS CLEARLY SHIT!
function mul!(y::ABV, U::UpperTriangular{T, <:BlockDiagonal{T}} where {T}, x::ABV)
    @assert are_conformal(U.data, x) && are_conformal(U.data, y)
    blocks = U.data.blocks.diag
    for r in 1:nblocks(U.data, 1)
        mul!(getblock(y, r), UpperTriangular(blocks[r]), getblock(x, r))
    end
    return y
end
mul!(y::ABV, U::UpperTriangular{<:Any, <:BlockDiagonal}, x::ABV) = mul!(y, U.data, x)
function mul!(y::ABV, D::BlockDiagonal{T, <:Diagonal{T}} where {T}, x::ABV)
    @assert are_conformal(D, x) && are_conformal(D, y)
    blocks = D.blocks.diag
    for r in 1:nblocks(D, 1)
        mul!(getblock(y, r), blocks[r], getblock(x, r))
    end
    return y
end
*(U::UpperTriangular{T, <:BlockDiagonal{T}} where {T}, x::ABV) = mul!(copy(x), U, x)

function mul!(Y::ABM, U::UpperTriangular{T, <:BlockDiagonal{T}} where {T}, X::ABM)
    @assert are_conformal(U.data, X) && are_conformal(U.data, Y)
    blocks = U.data.blocks.diag
    for r in 1:nblocks(U.data, 1)
        for c in 1:nblocks(X, 2)
            mul!(getblock(Y, r, c), blocks[r], getblock(X, r, c))
        end
    end
    return Y
end
function mul!(Y::ABM, U::UpperTriangular{T, <:Block})
*(U::UpperTriangular{T, <:BlockDiagonal{T}} where {T}, X::ABM) = mul!(copy(X), U, X)



################################## UpperTriangular BlockMatrices ###########################



# ################################# Symmetric BlockMatrices ##############################

# const BlockSymmetric{T, V} = Symmetric{T, <:BlockMatrix{T, V}}

# blocksizes(S::BlockSymmetric) = blocksizes(unbox(S))
# function getblock(X::BlockSymmetric, p::Int, q::Int)
#     @assert cumulsizes(X, 1) == cumulsizes(X, 2)
#     X_, uplo = unbox(X), X.uplo
#     if p < q
#         return uplo == 'U' ? getblock(X_, p, q) : transpose(getblock(X_, q, p))
#     elseif p == q
#         return Symmetric(getblock(X_, p, q))
#     else
#         return uplo == 'U' ? transpose(getblock(X_, q, p)) : getblock(X_, p, q)
#     end
# end



# ######################## Util for triangular block matrices ######################

# unbox(A::AbstractTriangular{T, <:ABM{T}} where T) = A.data
# blocksizes(A::AbstractTriangular{T, <:ABM{T}} where T) = blocksizes(unbox(A))
# function getblock(U::UpperTriangular{T, <:ABM{T}}, p::Int, q::Int) where T
#     @assert cumulsizes(U, 1) == cumulsizes(U, 2)
#     if p > q
#         return Zeros{T}(blocksize(U, (p, q)))
#     elseif p == q
#         return UpperTriangular(getblock(unbox(U), p, q))
#     else
#         return getblock(unbox(U), p, q)
#     end
# end
# function BlockMatrix(U::UpperTriangular{T, <:ABM{T}}) where T
#     B = similar(unbox(U))
#     for q in 1:nblocks(U, 2)
#         for p in 1:q-1
#             setblock!(B, getblock(U, p, q), p, q)
#         end
#         setblock!(B, UpperTriangular(getblock(U, q, q)), q, q)
#         for p in q+1:nblocks(U, 1)
#             setblock!(B, Zeros{T}(blocksize(U, (p, q))), p, q)
#         end
#     end
#     return B
# end

# function getblock(L::LowerTriangular{T, <:ABM{T}}, p::Int, q::Int) where T
#     @assert cumulsizes(L, 1) == cumulsizes(L, 2)
#     if p > q
#         return getblock(unbox(L), p, q)
#     elseif p == q
#         return LowerTriangular(getblock(unbox(L), p, q))
#     else
#         return Zeros{T}(blocksize(L, (p, q)))
#     end
# end
# function BlockMatrix(L::LowerTriangular{T, <:ABM{T}}) where T
#     B = similar(unbox(L))
#     for q in 1:nblocks(L, 2)
#         for p in 1:q-1
#             setblock!(B, Zeros{T}(blocksize(L, (p, q))), p, q)
#         end
#         setblock!(B, LowerTriangular(getblock(L, q, q)), q, q)
#         for p in q+1:nblocks(L, 1)
#             setblock!(B, getblock(L, p, q), p, q)
#         end
#     end
#     return B
# end



# ####################################### Copying ######################################

# copy(B::BlockSymmetric{T, <:ABM{T}} where T) = Symmetric(copy(unbox(B)))
# copy(L::LowerTriangular{T, <:BlockSymmetric{T}} where T) = LowerTriangular(copy(unbox(L)))
# copy(U::UpperTriangular{T, <:BlockSymmetric{T}} where T) = UpperTriangular(copy(unbox(U)))



# ####################################### Transposition ######################################

# unbox(A::AdjOrTrans) = A.parent
# blocksizes(A::AdjOrTrans) = BlockSizes(reverse(cumulsizes(unbox(A))))

# getblock(A::Adjoint, p::Int, q::Int) = getblock(A.parent, q, p)'
# getblock(A::Transpose, p::Int, q::Int) = transpose(getblock(A.parent, q, p))



# ####################################### Multiplication #####################################

"""
    are_conformal(A::BlockVecOrMat, B::BlockVecOrMat)

Test whether two block matrices (or vectors) are conformal. This criterion is stricter than
that for general matrices / vectors as we additionally require that each block be conformal
with block of the other matrix with which it will be multiplied. This ensures that the
result is itself straightforwardly representable as `BlockVecOrMat`.
"""
are_conformal(A::AVM, B::AVM) = cumulsizes(A, 2) == cumulsizes(B, 1)

# """
#     *(A::AdjTransTriABM, x::BlockVector)

# Matrix-vector multiplication between `BlockArray`s. Fails if block are not conformal.
# """
# function *(
#     A::Union{ABM{T}, BlockTri{T}, BlockAdjOrTrans{T}, AdjOrTrans{T, <:BlockTri{T}}},
#     x::ABV{T},
# ) where T
#     @assert are_conformal(A, x)
#     y = BlockVector{T}(undef_blocks, blocksizes(A, 1))
#     P, Q = nblocks(A)
#     for p in 1:P
#         setblock!(y, getblock(A, p, 1) * getblock(x, 1), p)
#         for q in 2:Q
#             setblock!(y, getblock(y, p) + getblock(A, p, q) * getblock(x, q), p)
#         end
#     end
#     return y
# end
# function *(A::AdjTransTriABM{T}, b::AV{T}) where {T}
#     @assert nblocks(A, 2) == 1
#     return A * BlockVector([b])
# end
# function *(A::AdjTransTriABM{T}, b::FillArrays.Zeros{T,1}) where T
#     return invoke(*, Tuple{AbstractMatrix, typeof(b)}, A, b)
# end



# """
#     *(A::AdjTransTriABM{T}, B::AdjTransTriABM{T}) where T

# Matrix-matrix multiplication between `BlockArray`s. Fails if blocks are not conformal.
# """
# function *(
#     A::Union{ABM{T}, BlockTri{T}, BlockAdjOrTrans{T}, AdjOrTrans{T, <:BlockTri{T}}},
#     B::Union{ABM{T}, BlockTri{T}, BlockAdjOrTrans{T}, AdjOrTrans{T, <:BlockTri{T}}},
# ) where T
#     @assert are_conformal(A, B)
#     C = BlockMatrix{T}(undef_blocks, blocksizes(A, 1), blocksizes(B, 2))
#     P, Q, R = nblocks(A, 1), nblocks(A, 2), nblocks(B, 2)
#     for p in 1:P, r in 1:R
#         setblock!(C, getblock(A, p, 1) * getblock(B, 1, r), p, r)
#         for q in 2:Q
#             setblock!(C, getblock(C, p, r) + getblock(A, p, q) * getblock(B, q, r), p, r)
#         end
#     end
#     return C
# end
# *(A::AdjTransTriABM{T}, B::AM{T}) where T = A * BlockMatrix([B])
# *(A::AM{T}, B::AdjTransTriABM{T}) where T = BlockMatrix([A]) * B

# # function *(A::LazyPDMat{T, <:Symmetric{T, <:ABM{T}}}, B::AdjTransTriABM{T}) where T
# #     return unbox(unbox(A)) * B
# # end

# const UpperOrAdjLower{T} = Union{
#     UpperTriangular{T, <:ABM{T}},
#     AdjOrTrans{T, <:LowerTriangular{T, <:ABM{T}}},
# }
# const LowerOrAdjUpper{T} = Union{
#     LowerTriangular{T, <:ABM{T}},
#     AdjOrTrans{T, <:UpperTriangular{T, <:ABM{T}}},
# }

# function \(U::UpperOrAdjLower{T}, x::ABV{T}) where T<:Real
#     @assert are_conformal(unbox(U), x)
#     y = BlockVector{T}(undef_blocks, blocksizes(U, 1))
#     for p in reverse(1:nblocks(y, 1))
#         setblock!(y, getblock(x, p), p)
#         for p′ in p+1:nblocks(y, 1)
#             setblock!(y, getblock(y, p) - getblock(U, p, p′) * getblock(y, p′), p)
#         end
#         setblock!(y, getblock(U, p, p) \ getblock(y, p), p)
#     end
#     return y
# end

# function _block_ldiv_mat_upper(U, X)
#     @assert are_conformal(unbox(U), X)
#     Y = BlockMatrix{eltype(U)}(undef_blocks, blocksizes(U, 1), blocksizes(X, 2))
#     for q in 1:nblocks(Y, 2), p in reverse(1:nblocks(Y, 1))
#         setblock!(Y, getblock(X, p, q), p, q)
#         for p′ in p+1:nblocks(Y, 1)
#             setblock!(Y, getblock(Y, p, q) - getblock(U, p, p′) * getblock(Y, p′, q), p, q)
#         end
#         setblock!(Y, getblock(U, p, p) \ getblock(Y, p, q), p, q)
#     end
#     return Y
# end
# function \(U::UpperOrAdjLower{T}, X::Union{ABM{T}, BlockTri{T}}) where T<:Real
#     return _block_ldiv_mat_upper(U, X)
# end
# function \(U::UpperOrAdjLower{T}, X::Adjoint{T, <:UpperTriangular{T, <:ABM{T}}}) where T<:Real
#     return _block_ldiv_mat_upper(U, X)
# end

# function \(L::LowerOrAdjUpper{T}, x::ABV{T}) where T<:Real
#     @assert are_conformal(unbox(L), x)
#     y = BlockVector{T}(undef_blocks, blocksizes(L, 1))
#     for p in 1:nblocks(y, 1)
#         setblock!(y, getblock(x, p), p)
#         for p′ in 1:p-1
#             setblock!(y, getblock(y, p) - getblock(L, p, p′) * getblock(y, p′), p)
#         end
#         setblock!(y, getblock(L, p, p) \ getblock(y, p), p)
#     end
#     return y
# end

# function _block_ldiv_mat_lower(L, X)
#     @assert are_conformal(unbox(L), X)
#     Y = BlockMatrix{eltype(L)}(undef_blocks, blocksizes(L, 1), blocksizes(X, 2))
#     for q in 1:nblocks(Y, 2), p in 1:nblocks(Y, 1)
#         setblock!(Y, getblock(X, p, q), p, q)
#         for p′ in 1:p-1
#             setblock!(Y, getblock(Y, p, q) - getblock(L, p, p′) * getblock(Y, p′, q), p, q)
#         end
#         setblock!(Y, getblock(L, p, p) \ getblock(Y, p, q), p, q)
#     end
#     return Y
# end
# function \(L::LowerOrAdjUpper{T}, X::Union{ABM{T}, BlockTri{T}}) where T<:Real
#     return _block_ldiv_mat_lower(L, X)
# end
# function \(L::LowerOrAdjUpper{T}, X::Adjoint{T, <:UpperTriangular{T, <:ABM{T}}}) where T<:Real
#     return _block_ldiv_mat_lower(L, X)
# end

# """
#     cholesky(A::Symmetric{T, <:BlockMatrix{T}}) where T<:Real

# Get the Cholesky decomposition of `A` in the form of a `BlockMatrix`.

# Only works for `A` where `is_block_symmetric(A) == true`. Assumes that we want the
# upper triangular version.
# """

# const BlockMaybeSymmetric{T, V} = Union{BlockMatrix{T, V}, BlockSymmetric{T, V}}

# # Compute the cholesky factorisation of a symmetric block matrix `A`, and a function to
# # compute the adjoint sensitivity.
# function cholesky_and_adjoint(A::BlockMaybeSymmetric{T, V}) where {T<:Real, V<:AM{T}}
#     U = BlockMatrix{T, V}(undef_blocks, blocksizes(A, 1), blocksizes(A, 2))
#     backs = Vector{Any}()

#     # Do an initial pass to fill each of the blocks with Zeros. This is cheap.
#     for q in 1:nblocks(U, 2), p in 1:nblocks(U, 1)
#         U[Block(p, q)] = Zeros{T}(blocksize(A, (p, q))...)
#     end

#     # Fill out the upper triangle with the Cholesky.
#     for j in 1:nblocks(A, 2)

#         # Update off-diagonals.
#         for i in 1:j-1
#             U[Block(i, j)] = copy(A[Block(i, j)])
#             for k in 1:i-1
#                 U[Block(i, j)] .-= U[Block(k, i)]' * U[Block(k, j)]
#             end
#             ldiv!(UpperTriangular(U[Block(i, i)])', U[Block(i, j)])
#         end

#         # Update diagonal.
#         U[Block(j, j)] = copy(A[Block(j, j)])
#         for k in 1:j-1
#             U[Block(j, j)] .-= U[Block(k, j)]' * U[Block(k, j)]
#         end
#         U[Block(j, j)] = cholesky(U[Block(j, j)]).U
#     end

#     return Cholesky(U, :U, 0), function(Δ)
#         Ā = BlockMatrix{T, V}(undef_blocks, blocksizes(A, 1), blocksizes(A, 1))
#         Ū = Δ.factors
#         for j in reverse(1:nblocks(A, 2))

#             Ā[Block(j, j)] = back(Ū[Block(j, j)])
#             for k in 1:j-1
#                 ĀjjUkj = Ā[Block(j, j)] * U[Block(k, j)]
#                 UkjĀjj = U[Block(k, j)] * Ā[Block(j, j)]
#                 Ū[Block(k, j)] .+= ĀjjUkj .+ UkjĀjj
#             end

#             for i in reverse(1:j-1)
#                 Ā[Block(i, j)] = Ā[Block(i, j)] + U[Block(i, i)] \ Ū[Block(i, j)]
#                 Ū[Block(i, i)] = Ū[Block(i, i)] - Ā[Block(i, j)] * U[Block(i, j)]'
#                 for k in 1:i-1
#                     Ū[Block(k, i)] = Ū[Block(k, i)] + Ā[Block(i, j)]' * U[Block(k, j)]
#                     Ū[Block(k, j)] = Ū[Block(k, j)] + U[Block(k, i)] * Ā[Block(i, j)]
#                 end
#             end
#         end
#         return (Ā,)
#     end
# end

# function cholesky(A::BlockMaybeSymmetric{T, V}) where {T<:Real, V<:AM{T}}
#     return cholesky_and_adjoint(A)[1]
# end

# # @adjoint function cholesky(A::BlockMaybeSymmetric{T, V}) where {T<:Real, V<:AM{T}}
# #     return cholesky(A), function(Δ)
# #         Ā = BlockMatrix{T, V}(undef_blocks, blocksizes(A, 1), blocksizes(A, 1))
# #         Ū = Δ.factors
# #         for j in reverse(1:nblocks(A, 2))
# #             Ā[Block(j, j)] = back(Ū[Block(j, j)])
# #             for k in 1:j-1
# #                 ĀjjUkj = Ā[Block(j, j)] * U[Block(k, j)]
# #                 UkjĀjj = U[Block(k, j)] * Ā[Block(j, j)]
# #                 Ū[Block(k, j)] .+= ĀjjUkj .+ UkjĀjj
# #             end

# #             for i in reverse(1:j-1)
# #                 Ā[Block(i, j)] = Ā[Block(i, j)] + U[Block(i, i)] \ Ū[Block(i, j)]
# #                 Ū[Block(i, i)] = Ū[Block(i, i)] - Ā[Block(i, j)] * U[Block(i, j)]'
# #                 for k in 1:i-1
# #                     Ū[Block(k, i)] = Ū[Block(k, i)] + Ā[Block(i, j)]' * U[Block(k, j)]
# #                     Ū[Block(k, j)] = Ū[Block(k, j)] + U[Block(k, i)] * Ā[Block(i, j)]
# #                 end
# #             end
# #         end
# #         return (Ā,)
# #     end
# # end

# function LinearAlgebra.diagzero(D::Diagonal{<:AM{T}}, r::Integer, c::Integer) where {T}
#     return Zeros{T}(size(D.diag[r], 1), size(D.diag[c], 2))
# end

# function cholesky(A::BlockMatrix{T, <:Diagonal{<:AbstractMatrix{T}}} where T)
#     Cs = [cholesky(A).U for A in diag(A.blocks)]
#     @show Cs, A.block_sizes
#     return Cholesky(BlockArrays._BlockArray(Diagonal(Cs), A.block_sizes), :U, 0)
# end

# # A slightly strange util function that shouldn't ever be used outside of `logdet`.
# reduce_diag(f, A::Matrix{T}) where {T<:Real} = sum(f, view(A, diagind(A)))
# function reduce_diag(f, A::BlockMatrix{T}) where T<:Real
#     return sum([reduce_diag(f, getblock(A, n, n)) for n in 1:nblocks(A, 1)])
# end

# logdet(C::Cholesky{<:Real, <:AbstractBlockMatrix}) = 2 * reduce_diag(log, C.factors)
# @adjoint function logdet(C::Cholesky{<:Real, <:AbstractBlockMatrix})
#     return logdet(C), function(Δ::Real)
#         function update_diag!(X::Matrix, A::Matrix)
#             X[diagind(X)] .= (2 * Δ) ./ A[diagind(A)]
#             return X
#         end
#         function update_diag!(X::BlockMatrix, A::BlockMatrix)
#             for n in 1:nblocks(A)[1]
#                 update_diag!(getblock(X, n, n), getblock(A, n, n))
#             end
#             return X
#         end
#         factors = update_diag!(zero(C.factors), C.factors)
#         return ((factors=factors, uplo=nothing, info=nothing),)
#     end
# end

# function +(u::UniformScaling, X::AbstractBlockMatrix)
#     @assert cumulsizes(X, 1) == cumulsizes(X, 2)
#     Y = copy(X)
#     for p in 1:nblocks(Y, 1)
#         setblock!(Y, getblock(Y, p, p) + u, p, p)
#     end
#     return Y
# end
# +(u::UniformScaling, X::Symmetric{T, <:ABM{T}} where T) = Symmetric(u + unbox(X))
# function +(X::AbstractBlockMatrix, u::UniformScaling)
#     @assert cumulsizes(X, 1) == cumulsizes(X, 2)
#     Y = copy(X)
#     for p in 1:nblocks(Y, 1)
#         setblock!(Y, u + getblock(Y, p, p), p, p)
#     end
#     return Y
# end
# +(X::Symmetric{T, <:ABM{T}} where T, u::UniformScaling) = Symmetric(unbox(X) + u)

# # Define addition and subtraction for compatible block matrices and vectors.
# import Base: +, -
# for foo in [:+, :-]
#     @eval function $foo(A::BV{T}, B::BV{T}) where T
#         @assert blocksizes(A) == blocksizes(B)
#         C = similar(A)
#         for p in 1:nblocks(C, 1)
#             setblock!(C, $foo(getblock(A, p), getblock(B, p)), p)
#         end
#         return C
#     end
#     @eval function $foo(A::BM{T}, B::BM{T}) where T
#         @assert blocksizes(A) == blocksizes(B)
#         C = similar(A)
#         for q in 1:nblocks(C, 2), p in 1:nblocks(C, 1)
#             setblock!(C, $foo(getblock(A, p, q), getblock(B, p, q)), p, q)
#         end
#         return C
#     end
# end



# #################################### Broadcasting ##########################################
# # Override the usual broadcasting machinery for BlockArrays. This is a pretty disgusting
# # hack. At the time of writing it, I was also writing my first year report, so was short of
# # time. This is generally an open problem for AbstractBlockArrays which really does need to
# # be resolved at some point.

# # Very specific `broadcast` method for particular use case. Needs to be generalised.
# function broadcasted(f, A::AbstractBlockVector, b::Real)
#     return BlockVector([broadcast(f, getblock(A, p), b) for p in 1:nblocks(A, 1)])
# end
# function broadcasted(f, A::AbstractBlockMatrix, b::Real)
#     return BlockMatrix([broadcast(f, getblock(A, p, q), b)
#         for p in 1:nblocks(A, 1), q in 1:nblocks(A, 2)])
# end
