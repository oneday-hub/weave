{-# LANGUAGE DataKinds #-}

module Main where

import Weave.Core
import Weave.Shape

main :: IO ()
main = do
  putStrLn "Weave DSL - Layer 1 + Layer 2 Demo"
  putStrLn ""

  -- Layer 1: expressions still work
  let e1 = lit 3.0 + lit 4.0 :: Expr Double
  putStrLn $ "3.0 + 4.0 = " ++ show (eval e1)
  putStrLn ""

  -- Layer 2: type-level shapes
  putStrLn "--- Layer 2: Type-Level Shapes ---"
  putStrLn ""

  -- vector
  let v = zeros :: Tensor '[5]
  putStrLn $ "zeros :: Tensor '[5]"
  putStrLn $ "  shape = " ++ show (tShape v)
  putStrLn $ "  size  = " ++ show (tSize v)
  putStrLn ""

  -- matrix
  let m = zeros :: Tensor '[3, 4]
  putStrLn $ "zeros :: Tensor '[3, 4]"
  putStrLn $ "  shape = " ++ show (tShape m)
  putStrLn $ "  size  = " ++ show (tSize m)
  putStrLn ""

  -- from list
  let t = fromList [1,2,3,4,5,6] :: Tensor '[2, 3]
  putStrLn $ "fromList [1..6] :: Tensor '[2, 3]"
  putStrLn $ "  shape  = " ++ show (tShape t)
  putStrLn $ "  values = " ++ show (toList t)
  putStrLn ""

  -- ones
  let o = ones :: Tensor '[2, 2]
  putStrLn $ "ones :: Tensor '[2, 2]"
  putStrLn $ "  values = " ++ show (toList o)
  putStrLn ""

  putStrLn "Layer 2 complete."