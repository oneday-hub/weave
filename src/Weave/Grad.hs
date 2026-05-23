{-# LANGUAGE ScopedTypeVariables #-}

-- |
-- Module   : Weave.Grad
-- Layer 4  : Automatic Differentiation
--
-- We use DUAL NUMBERS for forward mode autodiff.
-- A dual number D x dx carries:
--   x  = the value
--   dx = the derivative
--
-- Arithmetic on dual numbers applies the chain
-- rule automatically at every single operation.
-- No symbolic math. No approximation. Exact.

module Weave.Grad
  ( -- * Dual numbers
    Dual (..)
  , dualVal
  , dualDeriv
    -- * Differentiation
  , diff
  , diff2
    -- * Gradient computation
  , grad1D
  , gradVec
    -- * Gradient checking
  , checkGrad
  ) where

-- ── Dual Numbers ──────────────────────────────────────────

-- | A dual number: value paired with derivative.
-- D x dx means "value is x, derivative is dx"
--
-- How to use:
--   D 3.0 1.0  = the number 3, differentiating w.r.t. it
--   D 5.0 0.0  = the number 5, treated as a constant
data Dual a = D a a
  deriving (Eq)

instance Show a => Show (Dual a) where
  show (D x dx) =
    "D " ++ show x ++ " (deriv=" ++ show dx ++ ")"

-- | Extract the primal value from a dual number.
dualVal :: Dual a -> a
dualVal (D x _) = x

-- | Extract the derivative from a dual number.
dualDeriv :: Dual a -> a
dualDeriv (D _ dx) = dx

-- ── Arithmetic Instances ──────────────────────────────────
-- This is where the magic happens.
-- Each instance IS the chain rule for that operation.

instance Num a => Num (Dual a) where
  -- Sum rule: d/dx (f+g) = f' + g'
  (D x dx) + (D y dy) = D (x + y) (dx + dy)

  -- Difference rule: d/dx (f-g) = f' - g'
  (D x dx) - (D y dy) = D (x - y) (dx - dy)

  -- Product rule: d/dx (f*g) = f'g + fg'
  (D x dx) * (D y dy) = D (x * y) (dx*y + x*dy)

  -- Negate rule: d/dx (-f) = -f'
  negate (D x dx) = D (negate x) (negate dx)

  abs    (D x dx) = D (abs x) (dx * signum x)
  signum (D x _)  = D (signum x) 0
  fromInteger n   = D (fromInteger n) 0

instance Fractional a => Fractional (Dual a) where
  -- Quotient rule: d/dx (f/g) = (f'g - fg') / g^2
  (D x dx) / (D y dy) =
    D (x / y) ((dx*y - x*dy) / (y*y))
  fromRational r = D (fromRational r) 0

instance Floating a => Floating (Dual a) where
  pi            = D pi 0

  -- d/dx exp(f) = exp(f) * f'
  exp  (D x dx) = let ex = exp x in D ex (dx * ex)

  -- d/dx log(f) = f' / f
  log  (D x dx) = D (log x) (dx / x)

  -- d/dx sqrt(f) = f' / (2 * sqrt(f))
  sqrt (D x dx) = let sx = sqrt x in D sx (dx / (2 * sx))

  -- d/dx sin(f) = cos(f) * f'
  sin  (D x dx) = D (sin x) (dx * cos x)

  -- d/dx cos(f) = -sin(f) * f'
  cos  (D x dx) = D (cos x) (dx * (-sin x))

  -- d/dx tanh(f) = (1 - tanh^2(f)) * f'
  tanh (D x dx) =
    let tx = tanh x
    in D tx (dx * (1 - tx*tx))

  -- d/dx log(1 + e^f) softplus derivative

  asin (D x dx) = D (asin x) (dx / sqrt (1 - x*x))
  acos (D x dx) = D (acos x) ((-dx) / sqrt (1 - x*x))
  atan (D x dx) = D (atan x) (dx / (1 + x*x))
  sinh (D x dx) = D (sinh x) (dx * cosh x)
  cosh (D x dx) = D (cosh x) (dx * sinh x)
  asinh (D x dx) = D (asinh x) (dx / sqrt (1 + x*x))
  acosh (D x dx) = D (acosh x) (dx / sqrt (x*x - 1))
  atanh (D x dx) = D (atanh x) (dx / (1 - x*x))

-- ── Differentiation ───────────────────────────────────────

-- | Differentiate a scalar function at a point.
-- Injects x with tangent 1.0, reads derivative out.
--
-- Examples:
--   diff (\x -> x*x)       5.0  =  10.0  (2x at x=5)
--   diff (\x -> sin x)     0.0  =   1.0  (cos 0 = 1)
--   diff (\x -> exp x)     1.0  =   2.71 (exp 1)
--   diff (\x -> x*x + 3*x) 2.0  =   7.0  (2x+3 at x=2)
diff :: (Dual Double -> Dual Double) -> Double -> Double
diff f x = dualDeriv (f (D x 1.0))

-- | Second derivative via nested differentiation.
diff2 :: (Dual Double -> Dual Double) -> Double -> Double
diff2 f x =
  let h      = 1e-5
      fPrime = diff f
  in (fPrime (x + h) - fPrime (x - h)) / (2 * h)

-- ── Gradient Computation ──────────────────────────────────

-- | Gradient of a scalar function of one variable.
grad1D :: (Dual Double -> Dual Double) -> Double -> Double
grad1D = diff

-- | Gradient vector of a function R^n -> R.
-- Computes partial derivative w.r.t. each input.
-- Uses one forward pass per input dimension.
--
-- Example:
--   gradVec (\[x,y] -> x*x + y*y) [3.0, 4.0]
--   = [6.0, 8.0]   -- [2x, 2y] at (3,4)
--
--   gradVec (\[x,y] -> x*x + 2*y*y + x*y) [3.0, 4.0]
--   = [10.0, 19.0]  -- [2x+y, 4y+x] at (3,4)
gradVec :: ([Dual Double] -> Dual Double)
        -> [Double]
        -> [Double]
gradVec f xs =
  [ dualDeriv (f (oneHot i))
  | i <- [0..n-1]
  ]
  where
    n = length xs
    -- inject 1.0 in position i, 0.0 everywhere else
    oneHot i =
      [ D x (if j == i then 1.0 else 0.0)
      | (j, x) <- zip [0..] xs
      ]

-- ── Gradient Checking ─────────────────────────────────────

-- | Verify gradient against numerical finite differences.
-- f'(x) ~= (f(x+h) - f(x-h)) / (2h)
-- Returns True if analytic matches numeric within tolerance.
--
-- This is the standard sanity check in deep learning.
checkGrad :: (Double -> Double)   -- function
          -> (Double -> Double)   -- analytic derivative
          -> Double               -- point to check at
          -> Double               -- tolerance
          -> Bool
checkGrad f f' x tol =
  let h        = 1e-5
      numeric  = (f (x + h) - f (x - h)) / (2 * h)
      analytic = f' x
      delta    = abs (numeric - analytic)
  in delta < tol