{-# LANGUAGE GADTs                #-}
{-# LANGUAGE DataKinds            #-}
{-# LANGUAGE KindSignatures       #-}
{-# LANGUAGE TypeFamilies         #-}
{-# LANGUAGE TypeOperators        #-}
{-# LANGUAGE ScopedTypeVariables  #-}
{-# LANGUAGE FlexibleContexts     #-}
{-# LANGUAGE FlexibleInstances    #-}
{-# LANGUAGE UndecidableInstances #-}

-- |
-- Module   : Weave.Shape
-- Layer 2  : Type-Level Shape System
--
-- This is the core contribution of Weave.
-- Tensor shapes live in the TYPE not at runtime.
-- Shape mismatches are COMPILE TIME errors.
-- GHC is our shape checker.

module Weave.Shape
  ( Tensor (..)
  , KnownShape (..)
  , zeros
  , ones
  , fromList
  , toList
  , tShape
  , tSize
  ) where

import GHC.TypeLits
import Data.Proxy (Proxy (..))
import qualified Data.Vector as V

-- | A tensor indexed by its shape at the type level.
-- The shape is a TYPE LEVEL list of natural numbers.
--
-- Examples:
--   zeros :: Tensor '[3]      -- a vector  of 3 elements
--   zeros :: Tensor '[3, 4]   -- a matrix  of 3 rows, 4 cols
--   zeros :: Tensor '[2, 3, 4]-- a rank-3 tensor
--
-- The shape '[3, 4] is NOT a runtime value.
-- It lives in the TYPE -- GHC reads it at compile time.
data Tensor (shape :: [Nat]) =
  Tensor { tensorData :: V.Vector Double
         , tensorDims :: [Int]
         }

-- | Typeclass to convert type-level shapes to runtime lists.
-- This bridges the gap between compile time and runtime.
--
-- KnownShape '[3, 4] means GHC knows the shape [3, 4]
-- and can produce it as a runtime value [3, 4] when needed.
class KnownShape (s :: [Nat]) where
  shapeVal :: Proxy s -> [Int]

-- Base case: empty shape = scalar
instance KnownShape '[] where
  shapeVal _ = []

-- Recursive case: n : rest
instance (KnownNat n, KnownShape ns) => KnownShape (n ': ns) where
  shapeVal _ =
    fromIntegral (natVal (Proxy :: Proxy n))
    : shapeVal (Proxy :: Proxy ns)

-- ── Constructors ──────────────────────────────────────────

-- | Create a tensor of all zeros.
-- Shape is read from the RETURN TYPE by GHC.
--
-- Example:
--   let z = zeros :: Tensor '[3, 4]
--   tShape z  ==>  [3, 4]
--   tSize  z  ==>  12
zeros :: forall shape. KnownShape shape => Tensor shape
zeros =
  let dims = shapeVal (Proxy :: Proxy shape)
      size = product dims
  in Tensor (V.replicate size 0.0) dims

-- | Create a tensor of all ones.
ones :: forall shape. KnownShape shape => Tensor shape
ones =
  let dims = shapeVal (Proxy :: Proxy shape)
      size = product dims
  in Tensor (V.replicate size 1.0) dims

-- | Create a tensor from a flat list of Doubles.
-- List length must match the total size of the shape.
--
-- Example:
--   fromList [1,2,3,4,5,6] :: Tensor '[2, 3]
fromList :: forall shape. KnownShape shape
         => [Double] -> Tensor shape
fromList xs =
  let dims = shapeVal (Proxy :: Proxy shape)
      size = product dims
  in if length xs /= size
     then error $ "fromList: expected "
               ++ show size
               ++ " elements but got "
               ++ show (length xs)
     else Tensor (V.fromList xs) dims

-- | Extract values from a tensor as a flat list.
toList :: Tensor shape -> [Double]
toList = V.toList . tensorData

-- | Get the runtime shape of a tensor.
tShape :: Tensor shape -> [Int]
tShape = tensorDims

-- | Get total number of elements in a tensor.
tSize :: Tensor shape -> Int
tSize = V.length . tensorData

-- | Show instance for tensors
instance Show (Tensor shape) where
  show t =
    "Tensor " ++ show (tensorDims t)
    ++ " " ++ show (V.toList (tensorData t))