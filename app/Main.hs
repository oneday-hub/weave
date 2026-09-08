{-# LANGUAGE DataKinds #-}
module Main where

import Weave.Syntax

main :: IO ()
main = do

  putStrLn "========================================="
  putStrLn "  Weave DSL - C++ Style Arithmetic"
  putStrLn "========================================="
  putStrLn ""

  -- ── Scalar Arithmetic ────────────────────────
  -- C++: double a = 20.0;
  --      double b = 4.0;
  let a = auto (lit 20.0) :: Expr Double
  let b = auto (lit 4.0)  :: Expr Double

  putStrLn "--- Scalar Arithmetic ---"
  putStrLn ""

  -- C++: cout << "a + b" << compute(a+b);
  cout "  a        " (compute a)
  cout "  b        " (compute b)
  putStrLn ""
  cout "  a + b    " (compute (a + b))
  cout "  a - b    " (compute (a - b))
  cout "  a * b    " (compute (a * b))
  cout "  a / b    " (compute (a / b))
  putStrLn ""

  -- Nested
  putStrLn "--- Nested Expressions ---"
  putStrLn ""

  -- C++: auto e1 = (3.0 + 4.0) * 2.0;
  let e1 = auto ((lit 3.0 + lit 4.0) * lit 2.0)  :: Expr Double
  let e2 = auto (lit 10.0 / (lit 2.0 + lit 3.0)) :: Expr Double
  let e3 = auto ((lit 5.0 - lit 2.0) * (lit 6.0 + lit 1.0)) :: Expr Double

  cout "  (3+4)*2      " (compute e1)
  cout "  10/(2+3)     " (compute e2)
  cout "  (5-2)*(6+1)  " (compute e3)
  putStrLn ""

  -- ── Tensor Arithmetic ────────────────────────
  putStrLn "--- Tensor Arithmetic ---"
  putStrLn ""

  -- C++: Tensor<4> t1 = {10, 20, 30, 40};
  -- C++: Tensor<4> t2 = {2, 4, 5, 8};
  let t1 = makeTensor [10, 20, 30, 40] :: Tensor '[4]
  let t2 = makeTensor [2,  4,  5,  8 ] :: Tensor '[4]

  putStrLn $ "  Tensor<4> t1   = " ++ show (toList t1)
  putStrLn $ "  Tensor<4> t2   = " ++ show (toList t2)
  putStrLn ""

  -- C++: auto r1 = t1 |+| t2;
  let r1 = auto (t1 |+| t2)
  let r2 = auto (t1 |-| t2)
  let r3 = auto (t1 |*| t2)
  let r4 = auto (tScale 0.5 t1)

  putStrLn $ "  t1 |+| t2      = " ++ show (toList r1)
  putStrLn $ "  t1 |-| t2      = " ++ show (toList r2)
  putStrLn $ "  t1 |*| t2      = " ++ show (toList r3)
  putStrLn $ "  tScale(0.5,t1) = " ++ show (toList r4)
  putStrLn ""

  -- ── Matrix Arithmetic ────────────────────────
  putStrLn "--- Matrix Arithmetic ---"
  putStrLn ""

  -- C++: Tensor<2,2> M1 = {1,2,3,4};
  -- C++: Tensor<2,2> M2 = {5,6,7,8};
  let m1 = makeTensor [1,2,3,4] :: Tensor '[2,2]
  let m2 = makeTensor [5,6,7,8] :: Tensor '[2,2]

  putStrLn $ "  Tensor<2,2> M1  = " ++ show (toList m1)
  putStrLn $ "  Tensor<2,2> M2  = " ++ show (toList m2)
  putStrLn ""

  -- C++: auto add = M1 |+| M2;
  putStrLn $ "  M1 |+| M2       = " ++ show (toList (m1 |+| m2))
  putStrLn $ "  M1 |-| M2       = " ++ show (toList (m1 |-| m2))
  putStrLn $ "  M1 |*| M2       = " ++ show (toList (m1 |*| m2))

  -- C++: auto matprod = M1 |@| M2;
  let matprod = auto (m1 |@| m2)
  putStrLn $ "  M1 |@| M2       = " ++ show (toList matprod)
  putStrLn $ "  shape           = " ++ show (tShape matprod)
  putStrLn ""

  -- ── Activation Functions ─────────────────────
  putStrLn "--- Activation Functions ---"
  putStrLn ""

  -- C++: Tensor<5> v = {-2,-1,0,1,2};
  let v = makeTensor [-2,-1,0,1,2] :: Tensor '[5]
  putStrLn $ "  Tensor<5> v  = " ++ show (toList v)
  putStrLn ""

  -- C++: auto r = ReLU(v);
  putStrLn $ "  relu_(v)      = " ++ show (toList (relu_ v))
  putStrLn $ "  sigmoid_(v)   = " ++ show (map r4f (toList (sigmoid_ v)))
  putStrLn ""

  -- C++: Tensor<3> logits = {2,1,0.1};
  let logits = makeTensor [2.0, 1.0, 0.1] :: Tensor '[3]
  putStrLn $ "  logits       = " ++ show (toList logits)
  putStrLn $ "  softmax_(log) = " ++ show (map r4f (toList (softmax_ logits)))
  putStrLn $ "  sum          = " ++ show (sum_ (softmax_ logits))
  putStrLn ""

  -- ── Reductions ───────────────────────────────
  putStrLn "--- Reduction Functions ---"
  putStrLn ""

  -- C++: Tensor<5> nums = {1,2,3,4,5};
  let nums = makeTensor [1,2,3,4,5] :: Tensor '[5]
  putStrLn $ "  Tensor<5> nums = " ++ show (toList nums)
  putStrLn $ "  sum_(nums)     = " ++ show (sum_  nums)
  putStrLn $ "  mean_(nums)    = " ++ show (mean_ nums)
  putStrLn $ "  max_(nums)     = " ++ show (max_  nums)
  putStrLn ""

  -- ── Autodiff ─────────────────────────────────
  putStrLn "--- Automatic Differentiation ---"
  putStrLn ""

  -- C++: auto d1 = gradient([](auto x){ return x*x; }, 5.0);
  let d1 = gradient (\x -> x*x)       5.0
  let d2 = gradient (\x -> x*x*x)     3.0
  let d3 = gradient (\x -> sin x)     0.0
  let d4 = gradient (\x -> x*x + 3*x) 2.0

  putStrLn $ "  gradient(x^2,   5) = " ++ show d1
  putStrLn $ "  gradient(x^3,   3) = " ++ show d2
  putStrLn $ "  gradient(sin,   0) = " ++ show d3
  putStrLn $ "  gradient(x^2+3x,2) = " ++ show d4
  putStrLn ""

  -- C++: auto J = jacobian([](auto v){ return v[0]*v[0]+v[1]*v[1]; }, {3,4});
  let j1 = jacobian (\[x,y] -> x*x + y*y) [3.0, 4.0]
  putStrLn $ "  jacobian(x^2+y^2, [3,4]) = " ++ show j1
  putStrLn $ "  expected                 = [6.0, 8.0]"
  putStrLn ""

  putStrLn "========================================="
  putStrLn "  All operations complete."
  putStrLn "========================================="

r4f :: Double -> Double
r4f x = fromIntegral (round (x * 10000) :: Int) / 10000.0