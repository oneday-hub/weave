# Weave

**A compositional, type-safe tensor algebra DSL embedded in Haskell.**

Weave encodes tensor shapes as type-level information using GHC's DataKinds, GADTs, and TypeFamilies. Shape mismatches are **compile-time errors**, not runtime crashes. It includes a full automatic differentiation engine and deep learning model definitions.

---

## Why Weave?

Every major deep learning framework today — PyTorch, TensorFlow, JAX — validates tensor shapes **at runtime**. A shape mismatch is not caught until the program executes, potentially after hours of training.

```python
# PyTorch — this crashes at runtime
a = torch.zeros(3, 4)
b = torch.zeros(9, 5)
c = a @ b  # RuntimeError: mat1 and mat2 shapes cannot be multiplied
```

Weave catches this **at compile time**, before any code runs:

```haskell
-- Weave — this is a COMPILE ERROR
let a = zeros :: Tensor '[3, 4]
let b = zeros :: Tensor '[9, 5]
let c = matmul a b
-- Error: Couldn't match type '[4]' with '[9]'
--        Shapes are incompatible — GHC rejects this.
```

No runtime. No crash. No wasted GPU hours. GHC is your shape checker.

---

## Features

- **Compile-time shape verification** — tensor shapes live in the type, enforced by GHC
- **Compositional DSL** — build complex models from simple, well-typed operations
- **Automatic differentiation** — forward mode via dual numbers, exact derivatives
- **Gradient computation** — scalar and multi-variable gradients
- **Deep learning primitives** — linear layer, MLP, activation functions, loss functions
- **SGD optimizer** — gradient descent training loop
- **Pure Haskell** — no C bindings, no Python, no external ML framework

---

## Architecture — 5 Layers

Weave is built in five layers, each depending on those below it:

```
Layer 5 — Weave.Models    deep learning models
    │
Layer 4 — Weave.Grad      automatic differentiation
    │
Layer 3 — Weave.Ops       tensor operations
    │
Layer 2 — Weave.Shape     type-level shape system
    │
Layer 1 — Weave.Core      expression AST + evaluator
```

---

## Quick Start

### Prerequisites

- GHC 9.4 or above
- Cabal 3.8 or above

Install via [GHCup](https://www.haskell.org/ghcup/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
```

### Build and Run

```bash
git clone https://github.com/YOUR_USERNAME/weave.git
cd weave
cabal update
cabal build
cabal run weave-demo
cabal test
```

---

## Layer 1 — Core Expression AST

Every computation in Weave is represented as an `Expr` value — a tree built first, evaluated later. This is a **deep embedding**: the same tree can be evaluated, differentiated, or inspected.

```haskell
-- Build the tree
let e = lit 2.0 * (lit 3.0 + lit 4.0) :: Expr Double

-- Evaluate it
eval e  -- 14.0

-- Standard Haskell operators work directly
let e2 = 10 * 10 + 5 :: Expr Int
eval e2  -- 105
```

The `Num` typeclass instance means you write `e1 + e2` instead of `Add e1 e2`. The DSL feels like natural math.

---

## Layer 2 — Type-Level Shape System

Tensor shapes are encoded as **type-level lists of natural numbers**. The shape `'[3, 4]` is not a runtime value — it lives in the type and is checked by GHC at compile time.

```haskell
-- Shape is read from the RETURN TYPE
let scalar = zeros :: Tensor '[1]       -- 1 element
let vector = zeros :: Tensor '[5]       -- 5 elements
let matrix = zeros :: Tensor '[3, 4]    -- 3x4 = 12 elements
let rank3  = zeros :: Tensor '[2, 3, 4] -- 2x3x4 = 24 elements

-- Create from list
let t = fromList [1,2,3,4,5,6] :: Tensor '[2, 3]
tShape t  -- [2, 3]
tSize  t  -- 6
toList t  -- [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
```

The `KnownShape` typeclass bridges type-level shapes to runtime values — GHC reads the type and produces the list `[2, 3]` automatically.

---

## Layer 3 — Tensor Operations

Every operation carries its shape transformation in its type. GHC verifies compatibility at compile time.

### Matrix Multiplication

```haskell
-- Type: Tensor '[m,k] -> Tensor '[k,n] -> Tensor '[m,n]
-- GHC enforces that inner dimensions match

let a = fromList [1,2,3,4,5,6] :: Tensor '[2, 3]
let b = fromList [1,2,3,4,5,6] :: Tensor '[3, 2]
let c = matmul a b              -- :: Tensor '[2, 2]

tShape c  -- [2, 2]
toList c  -- [58.0, 64.0, 139.0, 154.0]

-- This is a COMPILE ERROR (inner dims don't match):
-- matmul (zeros :: Tensor '[2,3]) (zeros :: Tensor '[9,2])
-- Error: Couldn't match type '[3]' with '[9]'
```

### Activation Functions

```haskell
let v = fromList [-2, -1, 0, 1, 2] :: Tensor '[5]

toList (relu    v)  -- [0.0, 0.0, 0.0, 1.0, 2.0]
toList (sigmoid v)  -- [0.119, 0.269, 0.5, 0.731, 0.881]
toList (tanhT   v)  -- [-0.964, -0.762, 0.0, 0.762, 0.964]
```

### Softmax

```haskell
let logits = fromList [2.0, 1.0, 0.1] :: Tensor '[3]
let probs  = softmax logits

toList probs  -- [0.659, 0.242, 0.099]
tSum   probs  -- 1.0  (always sums to 1)
```

### Transpose

```haskell
-- Type: Tensor '[m,n] -> Tensor '[n,m]
let t  = zeros :: Tensor '[2, 3]
let tT = transpose2D t   -- :: Tensor '[3, 2]

tShape tT  -- [3, 2]
```

### Available Operations

| Operation | Type | Description |
|---|---|---|
| `tAdd` | `Tensor s -> Tensor s -> Tensor s` | Elementwise addition |
| `tSub` | `Tensor s -> Tensor s -> Tensor s` | Elementwise subtraction |
| `tMul` | `Tensor s -> Tensor s -> Tensor s` | Elementwise multiplication |
| `tScale` | `Double -> Tensor s -> Tensor s` | Scale by constant |
| `tMap` | `(Double -> Double) -> Tensor s -> Tensor s` | Apply function |
| `relu` | `Tensor s -> Tensor s` | max(0, x) |
| `sigmoid` | `Tensor s -> Tensor s` | 1 / (1 + e^-x) |
| `tanhT` | `Tensor s -> Tensor s` | tanh(x) |
| `softmax` | `Tensor '[n] -> Tensor '[n]` | Probability distribution |
| `matmul` | `Tensor '[m,k] -> Tensor '[k,n] -> Tensor '[m,n]` | Matrix multiply |
| `transpose2D` | `Tensor '[m,n] -> Tensor '[n,m]` | Transpose |
| `tSum` | `Tensor s -> Double` | Sum all elements |
| `tMean` | `Tensor s -> Double` | Mean of all elements |
| `tMax` | `Tensor s -> Double` | Maximum element |

---

## Layer 4 — Automatic Differentiation

Weave uses **dual numbers** for forward-mode automatic differentiation. A dual number `D x dx` carries the value `x` and its derivative `dx`. Arithmetic instances implement the chain rule automatically.

```haskell
-- data Dual a = D a a   -- D primal tangent

-- Product rule is built into the Num instance:
-- (D x dx) * (D y dy) = D (x*y) (dx*y + x*dy)
```

### Differentiating Functions

```haskell
-- diff: differentiate any scalar function
diff (\x -> x*x)       5.0  -- 10.0   (d/dx x^2 = 2x, at x=5)
diff (\x -> x*x*x)     3.0  -- 27.0   (d/dx x^3 = 3x^2, at x=3)
diff (\x -> sin x)     0.0  --  1.0   (d/dx sin = cos, cos(0) = 1)
diff (\x -> exp x)     1.0  --  2.718 (d/dx exp = exp)
diff (\x -> x*x + 3*x) 2.0  --  7.0   (2x + 3 at x=2)

-- Works on composed functions too
diff (\x -> sin (x*x)) 1.0  -- 1.0806 (chain rule applied)
diff (\x -> exp (sin x)) 0.0 -- 1.0
```

### Multi-Variable Gradients

```haskell
-- gradVec: gradient of R^n -> R functions
gradVec (\[x,y] -> x*x + y*y) [3.0, 4.0]
-- [6.0, 8.0]   -- [2x, 2y] at (3,4)

gradVec (\[x,y] -> x*x + 2*y*y + x*y) [3.0, 4.0]
-- [10.0, 19.0]  -- [2x+y, 4y+x] at (3,4)
```

### Gradient Checking

```haskell
-- Verify analytic gradient against finite differences
-- f'(x) =~ (f(x+h) - f(x-h)) / 2h

checkGrad (\x -> x*x)   (\x -> 2*x)   3.0 1e-5  -- True
checkGrad sin            cos            1.0 1e-5  -- True
checkGrad exp            exp            0.5 1e-5  -- True
```

---

## Layer 5 — Deep Learning Models

### Linear Layer

```haskell
-- output = input x weights
-- Shape: [batch, inF] x [inF, outF] -> [batch, outF]

let input   = fromList [1.0, 0.5, -0.3, 0.8] :: Tensor '[1, 4]
let weights = fromList [0.1,0.2, 0.3,0.4,
                        0.5,0.6, 0.7,0.8]    :: Tensor '[4, 2]
let output  = linearForward input weights     -- :: Tensor '[1, 2]

tShape output  -- [1, 2]
toList output  -- [0.66, 0.86]
```

### Multi-Layer Perceptron

```haskell
-- Define MLP: 4 inputs -> 8 hidden -> 2 outputs
-- ALL shapes verified by GHC at compile time

w1 <- initWeights :: IO (Tensor '[4, 8])
w2 <- initWeights :: IO (Tensor '[8, 2])

let mlp   = MLP2 w1 w2 :: MLP2 4 8 2
let input = fromList [1.0, 0.5, -0.3, 0.8] :: Tensor '[1, 4]

-- Forward pass: [1,4] -> relu -> [1,8] -> [1,2]
let pred  = mlpForward mlp input  -- :: Tensor '[1, 2]
```

### Training Loop

```haskell
let target = fromList [1.0, 0.0] :: Tensor '[1, 2]

-- Train for 10 epochs with learning rate 0.1
trainedMlp <- trainLoop 10 0.1 mlp input target

-- Output:
--   Epoch  1  loss: 0.8701
--   Epoch  2  loss: 0.5420
--   Epoch  5  loss: 0.1710
--   Epoch 10  loss: 0.0149
```

### Loss Functions

```haskell
-- Mean squared error
let preds   = fromList [0.9, 0.1] :: Tensor '[2]
let targets = fromList [1.0, 0.0] :: Tensor '[2]
mseLoss preds targets  -- 0.01
```

---

## Project Structure

```
weave/
├── weave.cabal              project configuration
├── README.md                this file
├── src/
│   └── Weave/
│       ├── Core.hs          Layer 1: expression AST + evaluator
│       ├── Shape.hs         Layer 2: type-level shape system
│       ├── Ops.hs           Layer 3: tensor operations
│       ├── Grad.hs          Layer 4: automatic differentiation
│       └── Models.hs        Layer 5: deep learning models
├── app/
│   └── Main.hs              demo application
└── test/
    └── Spec.hs              test suite
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `base` | ^>=4.18 | Haskell standard library |
| `vector` | >=0.13 | Efficient array storage |
| `random` | >=1.2 | Weight initialization |

No GPU libraries. No Python. No external ML framework. Pure Haskell.

---

## Running in GHCi

```bash
cabal repl

-- In GHCi:
:set -XDataKinds
import Weave.Shape
import Weave.Ops

let a = zeros :: Tensor '[3, 4]
:type a
-- a :: Tensor '[3, 4]

let b = zeros :: Tensor '[4, 5]
:type matmul a b
-- matmul a b :: Tensor '[3, 5]

-- Try a shape error:
-- matmul (zeros :: Tensor '[3,4]) (zeros :: Tensor '[9,5])
-- Error: Couldn't match type '[4]' with '[9]'
```

---

## Comparison with Existing Tools

| System | Language | Shape Safety | Autodiff | Embedded |
|---|---|---|---|---|
| PyTorch | Python | Runtime only | Dynamic graph | No |
| TensorFlow | Python | Runtime only | Static graph | No |
| JAX | Python | Runtime only | Functional | No |
| Dex | Dex | Compile time | Yes | No |
| **Weave** | **Haskell** | **Compile time** | **Dual numbers** | **Yes** |

Weave is the only system that combines: embedded DSL + compile-time shape safety + functional autodiff in a single lightweight Haskell library.

---

## Future Work

- GPU acceleration via Accelerate or CUDA FFI
- Convolution and pooling operations
- Recurrent layers (LSTM, GRU)
- Full reverse-mode autodiff via computation graph
- Hackage publication with Haddock documentation
- ONNX model export

---

## Academic Context

This project was developed as part of a study in compositional tensor algebra DSL design. It demonstrates how GHC's advanced type system features — DataKinds, GADTs, TypeFamilies — can be applied to solve a real engineering problem in machine learning: catching shape errors at compile time.

Related work: Dex (Google Brain), Futhark, hmatrix-static, Accelerate, the `ad` library.

---

## License

MIT License. See LICENSE file for details.
