{-# LANGUAGE DataKinds #-}
module Main where

import Weave.Core
import Weave.Shape
import Weave.Ops

main :: IO ()
main = do
  putStrLn "Weave DSL - Layer 3: Tensor Operations"
  putStrLn ""

  -- Elementwise operations
  putStrLn "--- Elementwise Operations ---"
  let a = fromList [1,2,3,4] :: Tensor '[4]
  let b = fromList [5,6,7,8] :: Tensor '[4]
  putStrLn $ "a         = " ++ show (toList a)
  putStrLn $ "b         = " ++ show (toList b)
  putStrLn $ "tAdd a b  = " ++ show (toList (tAdd a b))
  putStrLn $ "tSub a b  = " ++ show (toList (tSub a b))
  putStrLn $ "tMul a b  = " ++ show (toList (tMul a b))
  putStrLn $ "tScale 2  = " ++ show (toList (tScale 2.0 a))
  putStrLn ""

  -- Activations
  putStrLn "--- Activation Functions ---"
  let v = fromList [-2,-1,0,1,2] :: Tensor '[5]
  putStrLn $ "input     = " ++ show (toList v)
  putStrLn $ "relu      = " ++ show (toList (relu v))
  putStrLn $ "sigmoid   = " ++ show (map r4 (toList (sigmoid v)))
  putStrLn $ "tanhT     = " ++ show (map r4 (toList (tanhT v)))
  putStrLn ""

  -- Softmax
  putStrLn "--- Softmax ---"
  let logits = fromList [2.0, 1.0, 0.1] :: Tensor '[3]
  let probs  = softmax logits
  putStrLn $ "logits    = " ++ show (toList logits)
  putStrLn $ "probs     = " ++ show (map r4 (toList probs))
  putStrLn $ "sum       = " ++ show (tSum probs)
  putStrLn ""

  -- Matrix multiply
  putStrLn "--- Matrix Multiply ---"
  let m1 = fromList [1,2,3,4,5,6] :: Tensor '[2,3]
  let m2 = fromList [7,8,9,10,11,12] :: Tensor '[3,2]
  let m3 = matmul m1 m2
  putStrLn $ "A shape        = " ++ show (tShape m1)
  putStrLn $ "B shape        = " ++ show (tShape m2)
  putStrLn $ "matmul A B     = " ++ show (toList m3)
  putStrLn $ "output shape   = " ++ show (tShape m3)
  putStrLn ""

  -- Transpose
  putStrLn "--- Transpose ---"
  let t  = fromList [1,2,3,4,5,6] :: Tensor '[2,3]
  let tT = transpose2D t
  putStrLn $ "original shape = " ++ show (tShape t)
  putStrLn $ "transposed     = " ++ show (tShape tT)
  putStrLn ""

  putStrLn "Layer 3 complete."

r4 :: Double -> Double
r4 x = fromIntegral (round (x * 10000) :: Int) / 10000.0