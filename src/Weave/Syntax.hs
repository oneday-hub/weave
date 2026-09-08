{-# LANGUAGE DataKinds    #-}
{-# LANGUAGE TypeOperators #-}

-- |
-- Module   : Weave.Syntax
-- C++ style surface syntax for Weave DSL
-- Import this module to get C++-style names

module Weave.Syntax
  ( -- * C++ style types
    module Weave.Core
  , module Weave.Shape
  , module Weave.Ops
  , module Weave.Grad
    -- * C++ style constructors
  , auto
  , cout
    -- * C++ style tensor constructors
  , makeTensor
  , zeroTensor
  , oneTensor
    -- * C++ style operators
  , (|+|)
  , (|-|)
  , (|*|)
  , (|@|)
    -- * C++ style math functions
    , relu_
  , sigmoid_
  , softmax_
  , transpose_
  , matMul
    -- * C++ style reductions
  , sum_
  , mean_
  , max_
    -- * C++ style autodiff
  , gradient
  , jacobian
    -- * C++ style scalar eval
  , compute
  ) where

import Weave.Core
import Weave.Shape
import Weave.Ops
import Weave.Grad
import GHC.TypeLits

-- ── C++ style constructors ────────────────────────────────

-- | Like auto in C++ — just an identity alias
-- auto x = value   =>   let x = auto value
auto :: a -> a
auto = id

-- | Like cout << in C++ — just prints
cout :: Show a => String -> a -> IO ()
cout label val = putStrLn $ label ++ " = " ++ show val

-- | Like Tensor<n>{...} in C++
makeTensor :: KnownShape shape => [Double] -> Tensor shape
makeTensor = fromList

-- | Like Tensor<n> x = zeros() in C++
zeroTensor :: KnownShape shape => Tensor shape
zeroTensor = zeros

-- | Like Tensor<n> x = ones() in C++
oneTensor :: KnownShape shape => Tensor shape
oneTensor = ones

-- ── C++ style operators ───────────────────────────────────

-- | Like a + b in C++ (tensor addition)
(|+|) :: Tensor shape -> Tensor shape -> Tensor shape
(|+|) = tAdd
infixl 6 |+|

-- | Like a - b in C++ (tensor subtraction)
(|-|) :: Tensor shape -> Tensor shape -> Tensor shape
(|-|) = tSub
infixl 6 |-|

-- | Like a * b in C++ (elementwise multiply)
(|*|) :: Tensor shape -> Tensor shape -> Tensor shape
(|*|) = tMul
infixl 7 |*|

-- | Like a @ b in C++ / NumPy (matrix multiply)
(|@|) :: ( KnownNat m, KnownNat k, KnownNat n )
      => Tensor '[m, k]
      -> Tensor '[k, n]
      -> Tensor '[m, n]
(|@|) = matmul
infixl 7 |@|

-- ── C++ style function names ──────────────────────────────

-- | relu_(x) like C++ ReLU(x)
relu_ :: Tensor shape -> Tensor shape
relu_ = relu

-- | sigmoid_(x) like C++ Sigmoid(x)
sigmoid_ :: Tensor shape -> Tensor shape
sigmoid_ = sigmoid

-- | softmax_(x) like C++ Softmax(x)
softmax_ :: Tensor '[n] -> Tensor '[n]
softmax_ = softmax

-- | transpose_(x) like C++ Transpose(x)
transpose_ :: ( KnownNat m, KnownNat n )
           => Tensor '[m, n] -> Tensor '[n, m]
transpose_ = transpose2D

-- | matMul(a, b) like C++
matMul :: ( KnownNat m, KnownNat k, KnownNat n )
       => Tensor '[m, k]
       -> Tensor '[k, n]
       -> Tensor '[m, n]
matMul = matmul

-- ── C++ style reductions ──────────────────────────────────

-- | sum(x) like C++
sum_ :: Tensor shape -> Double
sum_ = tSum

-- | mean(x) like C++
mean_ :: Tensor shape -> Double
mean_ = tMean

-- | max(x) like C++
max_ :: Tensor shape -> Double
max_ = tMax

-- ── C++ style autodiff ────────────────────────────────────

-- | gradient(f, x) like C++
gradient :: (Dual Double -> Dual Double) -> Double -> Double
gradient = diff

-- | jacobian(f, xs) like C++
jacobian :: ([Dual Double] -> Dual Double) -> [Double] -> [Double]
jacobian = gradVec

-- ── C++ style eval ───────────────────────────────────────

-- | compute(expr) instead of eval(expr)
compute :: Expr a -> a
compute = eval