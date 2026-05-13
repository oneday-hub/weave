module Main where

import Weave.Core

main :: IO ()
main = do
  putStrLn "=== Weave Layer 1 Tests ==="
  putStrLn ""

  test "lit 42"           $ eval (lit 42 :: Expr Int)    == 42
  test "3.0 + 4.0 = 7.0"  $ eval (lit 3.0 + lit 4.0 :: Expr Double) == 7.0
  test "2.0 * 5.0 = 10.0" $ eval (lit 2.0 * lit 5.0 :: Expr Double) == 10.0
  test "10 - 3 = 7"       $ eval (lit 10 - lit 3 :: Expr Int) == 7
  test "negate 5 = -5"    $ eval (negate (lit 5) :: Expr Int) == -5
  test "2*(3+4) = 14"     $ eval (lit 2 * (lit 3 + lit 4) :: Expr Int) == 14

  putStrLn ""
  putStrLn "All Layer 1 tests done."

test :: String -> Bool -> IO ()
test name result =
  putStrLn $ (if result then "  [PASS]  " else "  [FAIL]  ") ++ name