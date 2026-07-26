{-# LANGUAGE DataKinds #-}

module Main where

import Weave.Core
import Weave.Ops
import Weave.Shape

import Control.Monad (when)
import System.Exit (exitFailure)

main :: IO ()
main = do
  putStrLn "=== Weave Tests ==="
  putStrLn ""

  results <- sequence
    [ test "lit 42"           $ eval (lit 42 :: Expr Int) == 42
    , test "3.0 + 4.0 = 7.0"  $ eval (lit 3.0 + lit 4.0 :: Expr Double) == 7.0
    , test "2.0 * 5.0 = 10.0" $ eval (lit 2.0 * lit 5.0 :: Expr Double) == 10.0
    , test "10 - 3 = 7"       $ eval (lit 10 - lit 3 :: Expr Int) == 7
    , test "negate 5 = -5"    $ eval (negate (lit 5) :: Expr Int) == -5
    , test "2*(3+4) = 14"     $ eval (lit 2 * (lit 3 + lit 4) :: Expr Int) == 14

    , test "tAdd [1,2,3] [4,5,6]" $
        toList (tAdd vectorA vectorB) == [5.0, 7.0, 9.0]
    , test "tSub [4,5,6] [1,2,3]" $
        toList (tSub vectorB vectorA) == [3.0, 3.0, 3.0]
    , test "tMul [1,2,3] [4,5,6]" $
        toList (tMul vectorA vectorB) == [4.0, 10.0, 18.0]
    , test "tScale 2 [1,-2,3]" $
        toList (tScale 2.0 signedVector) == [2.0, -4.0, 6.0]
    , test "relu [-1,0,2]" $
        toList (relu reluInput) == [0.0, 0.0, 2.0]
    , test "sigmoid [0]" $
        approxList (toList (sigmoid sigmoidInput)) [0.5]
    , test "softmax sums to 1" $
        approxEq (tSum (softmax logits)) 1.0
    , test "matmul [2,3] x [3,2]" $
        toList (matmul matrixA matrixB) == [58.0, 64.0, 139.0, 154.0]
    , test "transpose2D [2,3] -> [3,2]" $
        toList (transpose2D matrixA) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    , test "tSum [1,2,3,4]" $
        tSum reduceInput == 10.0
    , test "tMean [1,2,3,4]" $
        tMean reduceInput == 2.5
    , test "tMax [1,2,3,4]" $
        tMax reduceInput == 4.0
    ]

  putStrLn ""
  if and results
    then putStrLn "All tests passed."
    else do
      putStrLn "Some tests failed."
      exitFailure

test :: String -> Bool -> IO Bool
test name result =
  putStrLn ((if result then "  [PASS]  " else "  [FAIL]  ") ++ name)
    >> return result

approxEq :: Double -> Double -> Bool
approxEq a b = abs (a - b) < 1e-9

approxList :: [Double] -> [Double] -> Bool
approxList xs ys =
  length xs == length ys && and (zipWith approxEq xs ys)

vectorA :: Tensor '[3]
vectorA = fromList [1.0, 2.0, 3.0]

vectorB :: Tensor '[3]
vectorB = fromList [4.0, 5.0, 6.0]

signedVector :: Tensor '[3]
signedVector = fromList [1.0, -2.0, 3.0]

reluInput :: Tensor '[3]
reluInput = fromList [-1.0, 0.0, 2.0]

sigmoidInput :: Tensor '[1]
sigmoidInput = fromList [0.0]

logits :: Tensor '[3]
logits = fromList [1.0, 2.0, 3.0]

matrixA :: Tensor '[2, 3]
matrixA = fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

matrixB :: Tensor '[3, 2]
matrixB = fromList [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

reduceInput :: Tensor '[4]
reduceInput = fromList [1.0, 2.0, 3.0, 4.0]
