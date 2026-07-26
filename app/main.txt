{-# LANGUAGE DataKinds #-}
module Main where

import Weave.Core
import Weave.Shape
import Weave.Ops
import Weave.Grad
import Weave.Models

main :: IO ()
main = do
  putStrLn "Weave DSL - Layer 5: Deep Learning Models"
  putStrLn ""

  -- Linear layer
  putStrLn "--- Linear Layer ---"
  let input   = fromList [1.0, 0.5, -0.3, 0.8] :: Tensor '[1, 4]
  let weights = fromList [ 0.1, 0.2
                         , 0.3, 0.4
                         , 0.5, 0.6
                         , 0.7, 0.8 ] :: Tensor '[4, 2]
  let output  = linearForward input weights
  putStrLn $ "input   shape = " ++ show (tShape input)
  putStrLn $ "weights shape = " ++ show (tShape weights)
  putStrLn $ "output  shape = " ++ show (tShape output)
  putStrLn $ "output values = " ++ show (map r4 (toList output))
  putStrLn ""

  -- MLP
  putStrLn "--- MLP: 4 -> 8 -> 2 ---"
  putStrLn "Initializing weights..."
  w1init <- initWeights :: IO (Tensor '[4, 8])
  w2init <- initWeights :: IO (Tensor '[8, 2])
  let mlp0 = MLP2 w1init w2init :: MLP2 4 8 2
  putStrLn $ "w1 shape = " ++ show (tShape w1init)
  putStrLn $ "w2 shape = " ++ show (tShape w2init)

  let input2  = fromList [1.0, 0.5, -0.3, 0.8] :: Tensor '[1, 4]
  let target  = fromList [1.0, 0.0]             :: Tensor '[1, 2]
  let pred0   = mlpForward mlp0 input2
  let loss0   = mseLoss pred0 target
  putStrLn $ "initial prediction = " ++ show (map r4 (toList pred0))
  putStrLn $ "initial loss       = " ++ show (r4 loss0)
  putStrLn ""

  -- Training loop
  putStrLn "--- Training (SGD lr=0.1, 10 epochs) ---"
  putStrLn ""
  mlpFinal <- trainLoop 10 0.1 mlp0 input2 target
  putStrLn ""

  -- Final results
  let predF = mlpForward mlpFinal input2
  let lossF = mseLoss predF target
  putStrLn $ "final prediction = " ++ show (map r4 (toList predF))
  putStrLn $ "final loss       = " ++ show (r4 lossF)
  putStrLn ""

  -- Loss function demo
  putStrLn "--- MSE Loss ---"
  let p = fromList [0.9, 0.1] :: Tensor '[2]
  let t = fromList [1.0, 0.0] :: Tensor '[2]
  putStrLn $ "predictions = " ++ show (toList p)
  putStrLn $ "targets     = " ++ show (toList t)
  putStrLn $ "MSE loss    = " ++ show (mseLoss p t)
  putStrLn ""

  putStrLn "Layer 5 complete."
  putStrLn ""
  putStrLn "ALL 5 LAYERS COMPLETE - Weave DSL is built!"

r4 :: Double -> Double
r4 x = fromIntegral (round (x * 10000) :: Int) / 10000.0