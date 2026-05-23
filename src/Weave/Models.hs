{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE KindSignatures      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}

-- |
-- Module   : Weave.Models
-- Layer 5  : Deep Learning Model Definitions
--
-- This is the top layer of Weave.
-- Everything from layers 1 to 4 comes together here.
-- Models are pure functions over typed tensors.
-- All shapes verified at compile time.

module Weave.Models
  ( -- * Linear layer
    linearForward
    -- * MLP
  , MLP2 (..)
  , mlpForward
    -- * Loss functions
  , mseLoss
    -- * Optimizer
  , sgdStep
    -- * Training
  , initWeights
  , trainStep
  , trainLoop
  ) where

import Weave.Shape
import Weave.Ops
import Weave.Grad
import GHC.TypeLits
import Data.Proxy (Proxy (..))
import qualified Data.Vector as V
import System.Random (randomRIO)

-- ── Linear Layer ──────────────────────────────────────────

-- | A fully connected linear layer.
-- output = input x weights
--
-- Type guarantees:
--   input  :: Tensor '[batch, inF]
--   weights:: Tensor '[inF,  outF]
--   output :: Tensor '[batch, outF]
--
-- GHC verifies input columns = weight rows.
linearForward
  :: ( KnownNat batch
     , KnownNat inF
     , KnownNat outF
     )
  => Tensor '[batch, inF]
  -> Tensor '[inF,   outF]
  -> Tensor '[batch, outF]
linearForward input weights = matmul input weights

-- ── MLP: 2-layer Neural Network ───────────────────────────

-- | A 2-layer MLP with shapes fixed in the type.
-- Network: input -> hidden (ReLU) -> output
--
-- The record type itself encodes the architecture.
-- You cannot accidentally swap w1 and w2 --
-- GHC would reject it as a type error.
data MLP2 (inF :: Nat) (hidF :: Nat) (outF :: Nat) =
  MLP2
    { w1 :: Tensor '[inF,  hidF]   -- layer 1 weights
    , w2 :: Tensor '[hidF, outF]   -- layer 2 weights
    }

-- | Forward pass through the 2-layer MLP.
-- Shape flow:
--   input [b, inF]
--   -> matmul w1 -> [b, hidF]
--   -> relu       -> [b, hidF]
--   -> matmul w2 -> [b, outF]
mlpForward
  :: ( KnownNat b
     , KnownNat inF
     , KnownNat hidF
     , KnownNat outF
     )
  => MLP2 inF hidF outF
  -> Tensor '[b, inF]
  -> Tensor '[b, outF]
mlpForward mlp input =
  let h1 = relu (linearForward input (w1 mlp))
      h2 = linearForward h1 (w2 mlp)
  in  h2

-- ── Loss Functions ────────────────────────────────────────

-- | Mean squared error loss.
-- Measures how far predictions are from targets.
-- Both must have identical shapes.
mseLoss :: Tensor shape -> Tensor shape -> Double
mseLoss preds targets =
  let diff = tSub preds targets
      sq   = tMul diff diff
  in  tMean sq

-- ── SGD Optimizer ─────────────────────────────────────────

-- | One step of stochastic gradient descent.
-- new_params = params - learning_rate * gradients
sgdStep
  :: Double         -- learning rate
  -> Tensor shape   -- current parameters
  -> Tensor shape   -- gradients
  -> Tensor shape   -- updated parameters
sgdStep lr params grads =
  tSub params (tScale lr grads)

-- ── Weight Initialization ─────────────────────────────────

-- | Initialize weights with small random values.
-- Random values in range (-0.5, 0.5).
initWeights
  :: forall shape. KnownShape shape
  => IO (Tensor shape)
initWeights = do
  let dims = shapeVal (Proxy :: Proxy shape)
      size = product dims
  vals <- mapM (\_ -> randomRIO (-0.5, 0.5)) [1..size]
  return $ Tensor (V.fromList vals) dims

-- ── Numerical Gradient ────────────────────────────────────

-- | Compute gradient of a weight tensor numerically.
-- Uses finite differences: (f(w+h) - f(w-h)) / 2h
-- This is how we do the backward pass in this version.
numericalGrad
  :: KnownShape shape
  => (Tensor shape -> Double)
  -> Tensor shape
  -> Tensor shape
numericalGrad lossF (Tensor v dims) =
  let h     = 1e-4
      n     = V.length v
      grads = V.generate n $ \i ->
                let vPlus  = v V.// [(i, (v V.! i) + h)]
                    vMinus = v V.// [(i, (v V.! i) - h)]
                    lPlus  = lossF (Tensor vPlus  dims)
                    lMinus = lossF (Tensor vMinus dims)
                in  (lPlus - lMinus) / (2 * h)
  in Tensor grads dims

-- ── Training ──────────────────────────────────────────────

-- | One training step:
-- 1. Forward pass  -> compute prediction
-- 2. Loss          -> measure error
-- 3. Gradients     -> compute dL/dW
-- 4. SGD step      -> update weights
trainStep
  :: ( KnownNat inF
     , KnownNat hidF
     , KnownNat outF
     , KnownShape '[inF,  hidF]
     , KnownShape '[hidF, outF]
     )
  => Double
  -> MLP2 inF hidF outF
  -> Tensor '[1, inF]
  -> Tensor '[1, outF]
  -> (MLP2 inF hidF outF, Double)
trainStep lr mlp input target =
  let pred   = mlpForward mlp input
      loss   = mseLoss pred target
      lossW1 w = mseLoss (mlpForward (mlp { w1 = w }) input) target
      lossW2 w = mseLoss (mlpForward (mlp { w2 = w }) input) target
      gradW1 = numericalGrad lossW1 (w1 mlp)
      gradW2 = numericalGrad lossW2 (w2 mlp)
      newW1  = sgdStep lr (w1 mlp) gradW1
      newW2  = sgdStep lr (w2 mlp) gradW2
  in  (MLP2 newW1 newW2, loss)

-- | Run training for N epochs.
-- Prints loss at each epoch.
trainLoop
  :: ( KnownNat inF
     , KnownNat hidF
     , KnownNat outF
     , KnownShape '[inF,  hidF]
     , KnownShape '[hidF, outF]
     )
  => Int
  -> Double
  -> MLP2 inF hidF outF
  -> Tensor '[1, inF]
  -> Tensor '[1, outF]
  -> IO (MLP2 inF hidF outF)
trainLoop 0 _ mlp _ _ = return mlp
trainLoop epochs lr mlp input target = do
  let (mlp', loss) = trainStep lr mlp input target
      epoch        = 11 - epochs
  putStrLn $ "  Epoch " ++ pad epoch
          ++ "  loss: "  ++ showLoss loss
  trainLoop (epochs - 1) lr mlp' input target
  where
    pad n     = replicate (2 - length (show n)) ' ' ++ show n
    showLoss l = show (fromIntegral
                  (round (l * 10000) :: Int) / 10000.0 :: Double)