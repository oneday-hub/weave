module Main where

import Weave.Core

main :: IO ()
main = do
  putStrLn "Weave DSL - Basic Arithmetic"
  putStrLn ""

  -- Addition
  let a = lit 10.0 + lit 5.0 :: Expr Double
  putStrLn $ "10 + 5  = " ++ show (eval a)

  -- Subtraction
  let s = lit 10.0 - lit 5.0 :: Expr Double
  putStrLn $ "10 - 5  = " ++ show (eval s)

  -- Multiplication
  let m = lit 10.0 * lit 5.0 :: Expr Double
  putStrLn $ "10 * 5  = " ++ show (eval m)

  -- Division
  let d = lit 10.0 / lit 5.0 :: Expr Double
  putStrLn $ "10 / 5  = " ++ show (eval d)

  putStrLn ""
  putStrLn "Done."