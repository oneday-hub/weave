{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE TypeFamilies        #-}
{-# LANGUAGE TypeOperators       #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}

-- |
-- Module   : Weave.Ops
-- Layer 3  : Shape-Safe Tensor Operations
--
-- Every operation carries its shape transformation
-- in its type signature. GHC verifies compatibility.
-- Incompatible shapes = compile time error.

module Weave.Ops
  ( -- * Elementwise operations
    tAdd
  , tSub
  , tMul
  , tScale
  , tMap
    -- * Activation functions
  , relu
  , sigmoid
  , tanhT
  , softmax
    -- * Matrix operations
  , matmul
  , transpose2D
    -- * Reductions
  , tSum
  , tMean
  , tMax
  ) where

import Weave.Shape
import GHC.TypeLits
import Data.Proxy (Proxy (..))
import qualified Data.Vector as V

-- ── Elementwise Operations ────────────────────────────────
-- These preserve shape: input shape = output shape

-- | Elementwise addition.
-- Both tensors MUST have identical shapes.
-- Different shapes = compile time type error.
tAdd :: Tensor shape -> Tensor shape -> Tensor shape
tAdd (Tensor a dims) (Tensor b _) =
  Tensor (V.zipWith (+) a b) dims

-- | Elementwise subtraction.
tSub :: Tensor shape -> Tensor shape -> Tensor shape
tSub (Tensor a dims) (Tensor b _) =
  Tensor (V.zipWith (-) a b) dims

-- | Elementwise multiplication (Hadamard product).
tMul :: Tensor shape -> Tensor shape -> Tensor shape
tMul (Tensor a dims) (Tensor b _) =
  Tensor (V.zipWith (*) a b) dims

-- | Scale every element by a constant.
tScale :: Double -> Tensor shape -> Tensor shape
tScale s (Tensor v dims) =
  Tensor (V.map (* s) v) dims

-- | Apply any function to every element.
tMap :: (Double -> Double) -> Tensor shape -> Tensor shape
tMap f (Tensor v dims) =
  Tensor (V.map f v) dims

-- ── Activation Functions ──────────────────────────────────
-- All activations preserve shape

-- | ReLU: max(0, x) elementwise.
-- Negative values become 0, positive unchanged.
relu :: Tensor shape -> Tensor shape
relu = tMap (max 0)

-- | Sigmoid: 1 / (1 + e^-x) elementwise.
-- Squashes values to range (0, 1).
sigmoid :: Tensor shape -> Tensor shape
sigmoid = tMap (\x -> 1.0 / (1.0 + exp (-x)))

-- | Tanh activation elementwise.
-- Squashes values to range (-1, 1).
tanhT :: Tensor shape -> Tensor shape
tanhT = tMap tanh

-- | Softmax over a 1D tensor.
-- Converts logits to probabilities that sum to 1.0
softmax :: Tensor '[n] -> Tensor '[n]
softmax (Tensor v dims) =
  let maxV = V.maximum v
      exps = V.map (\x -> exp (x - maxV)) v
      sumE = V.sum exps
  in Tensor (V.map (/ sumE) exps) dims

-- ── Matrix Operations ─────────────────────────────────────

-- | Matrix multiplication.
-- Type enforces: [m,k] x [k,n] -> [m,n]
-- Inner dimensions MUST match.
-- This is verified by GHC at compile time.
--
-- Example:
--   let a = zeros :: Tensor '[2, 3]
--   let b = zeros :: Tensor '[3, 5]
--   let c = matmul a b  -- :: Tensor '[2, 5]
--
-- This would be a COMPILE ERROR:
--   let bad = matmul (zeros :: Tensor '[2,3])
--                    (zeros :: Tensor '[9,5])
--   -- Error: could not match '[3] with '[9]'
matmul :: forall m k n.
          ( KnownNat m
          , KnownNat k
          , KnownNat n
          )
       => Tensor '[m, k]
       -> Tensor '[k, n]
       -> Tensor '[m, n]
matmul (Tensor a _) (Tensor b _) =
  let m   = fromIntegral (natVal (Proxy :: Proxy m))
      k   = fromIntegral (natVal (Proxy :: Proxy k))
      n   = fromIntegral (natVal (Proxy :: Proxy n))
      res = V.generate (m * n) $ \idx ->
              let row = idx `div` n
                  col = idx `mod` n
              in sum [ (a V.! (row * k + i))
                     * (b V.! (i  * n + col))
                     | i <- [0..k-1] ]
  in Tensor res [m, n]

-- | Transpose a 2D matrix.
-- Type: Tensor '[m,n] -> Tensor '[n,m]
transpose2D :: forall m n.
               ( KnownNat m
               , KnownNat n
               )
            => Tensor '[m, n]
            -> Tensor '[n, m]
transpose2D (Tensor v _) =
  let m   = fromIntegral (natVal (Proxy :: Proxy m))
      n   = fromIntegral (natVal (Proxy :: Proxy n))
      res = V.generate (m * n) $ \idx ->
              let row = idx `div` m
                  col = idx `mod` m
              in v V.! (col * n + row)
  in Tensor res [n, m]

-- ── Reductions ────────────────────────────────────────────
-- These collapse all elements to a single scalar

-- | Sum all elements.
tSum :: Tensor shape -> Double
tSum (Tensor v _) = V.sum v

-- | Mean of all elements.
tMean :: Tensor shape -> Double
tMean (Tensor v _) = V.sum v / fromIntegral (V.length v)

-- | Maximum element value.
tMax :: Tensor shape -> Double
tMax (Tensor v _) = V.maximum v