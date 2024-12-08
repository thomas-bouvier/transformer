# transformer

We build a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3.

Some additional resources on transformers:

- [Creating a Transformer From Scratch, Part One: The Attention Mechanism](https://benjaminwarner.dev/2023/07/01/attention-mechanism)
- [Creating a Transformer From Scratch, Part Two: The Rest of the Transformer](https://benjaminwarner.dev/2023/07/28/rest-of-the-transformer)
- [[M2L 2024] Transformers - Lucas Beyer](https://www.youtube.com/watch?v=bMXqnLiVgLk)

Training on a single A100 40GB takes ~15 minutes and produces the text in `output.txt`. The model seems to overfit quite a bit, possibly because of the limited training data.

```
Step 0: train loss 4.475174903869629, val loss 4.470689296722412
Step 500: train loss 1.7102497816085815, val loss 1.863983154296875
Step 1000: train loss 1.444942593574524, val loss 1.6555002927780151
Step 1500: train loss 1.3290531635284424, val loss 1.5621485710144043
Step 2000: train loss 1.2531943321228027, val loss 1.5290812253952026
Step 2500: train loss 1.1971135139465332, val loss 1.5185022354125977
Step 3000: train loss 1.1427865028381348, val loss 1.5040178298950195
Step 3500: train loss 1.0996745824813843, val loss 1.5207672119140625
Step 4000: train loss 1.0571194887161255, val loss 1.5238550901412964
Step 4500: train loss 1.0115548372268677, val loss 1.524843692779541
```

## Setup

Follow the instructions from [`devenv`](https://devenv.sh/getting-started/)
using the instructions found here.

### Install `nix`

```bash
### Via https://zero-to-nix.com/start/install (recommended)
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install

### Via https://devenv.sh/getting-started/
## Linux
sh <(curl -L https://nixos.org/nix/install) --daemon

## macOS
curl -L https://raw.githubusercontent.com/NixOS/experimental-nix-installer/main/nix-installer.sh | sh -s install

## WSL2
sh <(curl -L https://nixos.org/nix/install) --no-daemon
```

### Install `devenv`

```bash
## General
nix-env -iA devenv -f https://github.com/NixOS/nixpkgs/tarball/nixpkgs-unstable

## NixOS
# Add the following to your configuration.nix somewhere
environment.systemPackages = [ 
  pkgs.devenv
];
```

#### `devenv.nix`

Defines the configuration for the `devenv` shell. This is where we define all
the tooling, packages, scripts, services, processes, etc. that we need for the
project.

#### `devenv.yaml`

The `yaml` defines the sources for all the packages, i.e. where are we getting
the cached builds or build instructions for `nix`.

#### TODO

CUDA support:

- https://github.com/johnrizzo1/myada/blob/6928288910bfd1df8993d8c61bdc5d24d92b4c9e/devenv.nix
- https://github.com/borh/dm-annotations/blob/06035f4547c68b7bf03b757215b48c76568d8d15/devenv.nix
