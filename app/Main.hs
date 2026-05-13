module Main where

import Weave.Core

main :: IO ()
main = do
  putStrLn "Weave DSL - Layer 1 Demo"
  putStrLn ""

  -- Basic addition
  let e1 = lit 3.0 + lit 4.0 :: Expr Double
  putStrLn $ "3.0 + 4.0          = " ++ show (eval e1)

  -- Nested expression
  let e2 = lit 2.0 * (lit 5.0 - lit 1.0) :: Expr Double
  putStrLn $ "2.0 * (5.0 - 1.0)  = " ++ show (eval e2)

  -- Negation
  let e3 = negate (lit 7.0) + lit 10.0 :: Expr Double
  putStrLn $ "negate 7.0 + 10.0  = " ++ show (eval e3)

  -- Integer expressions
  let e4 = 10 * 10 + 5 :: Expr Int
  putStrLn $ "10 * 10 + 5        = " ++ show (eval e4)

  putStrLn ""
  putStrLn "Layer 1 complete."