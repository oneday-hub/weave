{-# LANGUAGE GADTs #-}

-- |
-- Module    : Weave.Core
-- Layer 1   : Core Expression AST and Evaluator
--
-- Every computation in Weave is an Expr value.
-- We build the tree first, evaluate it later.
-- This is called a DEEP EMBEDDING.

module Weave.Core
  ( Expr (..)
  , eval
  , lit
  ) where

-- | The expression tree type.
-- Expr a = a computation that produces a value of type a
--
-- Examples:
--   Lit 3.0                        -- just the number 3.0
--   Add (Lit 3.0) (Lit 4.0)        -- 3.0 + 4.0
--   Mul (Lit 2.0) (Add (Lit 3.0) (Lit 4.0))  -- 2.0 * (3.0 + 4.0)

data Expr a where
  Lit :: a                         -> Expr a
  Add :: Num a => Expr a -> Expr a -> Expr a
  Sub :: Num a => Expr a -> Expr a -> Expr a
  Mul :: Num a => Expr a -> Expr a -> Expr a
  Neg :: Num a => Expr a           -> Expr a

-- | Smart constructor: lift a plain value into the DSL
lit :: a -> Expr a
lit = Lit

-- | Evaluate an expression tree and return its result.
-- Walks the AST node by node and computes the value.
eval :: Expr a -> a
eval (Lit x)   = x
eval (Add a b) = eval a + eval b
eval (Sub a b) = eval a - eval b
eval (Mul a b) = eval a * eval b
eval (Neg a)   = negate (eval a)

-- | Allows writing  e1 + e2  instead of  Add e1 e2
-- Standard Haskell operators now work on DSL expressions
instance Num a => Num (Expr a) where
  (+)         = Add
  (-)         = Sub
  (*)         = Mul
  negate      = Neg
  abs _       = error "abs not supported yet"
  signum _    = error "signum not supported yet"
  fromInteger = Lit . fromInteger

instance Show a => Show (Expr a) where
  show expr = "Expr => " ++ show (eval expr)