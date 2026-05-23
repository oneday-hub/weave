{-# LANGUAGE DataKinds #-}
module Main where

import Weave.Core
import Weave.Shape
import Weave.Ops
import Weave.Grad

main :: IO ()
main = do
  putStrLn "Weave DSL - Layer 4: Automatic Differentiation"
  putStrLn ""

  -- Basic derivatives
  putStrLn "--- Forward Mode: diff ---"
  putStrLn $ "d/dx x^2    at x=5   = " ++ show (diff (\x -> x*x) 5.0)
  putStrLn $ "d/dx x^3    at x=3   = " ++ show (diff (\x -> x*x*x) 3.0)
  putStrLn $ "d/dx sin(x) at x=0   = " ++ show (r4 (diff sin 0.0))
  putStrLn $ "d/dx exp(x) at x=1   = " ++ show (r4 (diff exp 1.0))
  putStrLn $ "d/dx x^2+3x at x=2   = " ++ show (diff (\x -> x*x + 3*x) 2.0)
  putStrLn ""

  -- Nested functions
  putStrLn "--- Composed Functions ---"
  putStrLn $ "d/dx sin(x^2)  at x=1 = " ++ show (r6 (diff (\x -> sin (x*x)) 1.0))
  putStrLn $ "d/dx exp(sin x) at x=0 = " ++ show (r6 (diff (\x -> exp (sin x)) 0.0))
  putStrLn ""

  -- Gradient vector
  putStrLn "--- Gradient Vector ---"
  let g1 = gradVec (\[x,y] -> x*x + y*y) [3.0, 4.0]
  putStrLn $ "f(x,y) = x^2 + y^2"
  putStrLn $ "grad at (3,4) = " ++ show g1
  putStrLn $ "expected      = [6.0, 8.0]"
  putStrLn ""

  let g2 = gradVec (\[x,y] -> x*x + 2*y*y + x*y) [3.0, 4.0]
  putStrLn $ "f(x,y) = x^2 + 2y^2 + xy"
  putStrLn $ "grad at (3,4) = " ++ show (map r4 g2)
  putStrLn $ "expected      = [10.0, 19.0]"
  putStrLn ""

  -- Gradient checks
  putStrLn "--- Gradient Checks (analytic vs finite diff) ---"
  let ok1 = checkGrad (\x -> x*x)   (\x -> 2*x)   3.0 1e-5
  let ok2 = checkGrad (\x -> x*x*x) (\x -> 3*x*x) 2.0 1e-5
  let ok3 = checkGrad sin            cos            1.0 1e-5
  let ok4 = checkGrad exp            exp            0.5 1e-5
  putStrLn $ "x^2   at x=3: " ++ result ok1
  putStrLn $ "x^3   at x=2: " ++ result ok2
  putStrLn $ "sin   at x=1: " ++ result ok3
  putStrLn $ "exp   at x=0.5: " ++ result ok4
  putStrLn ""

  putStrLn "Layer 4 complete."

result :: Bool -> String
result True  = "PASS"
result False = "FAIL"

r4 :: Double -> Double
r4 x = fromIntegral (round (x * 10000) :: Int) / 10000.0

r6 :: Double -> Double
r6 x = fromIntegral (round (x * 1000000) :: Int) / 1000000.0