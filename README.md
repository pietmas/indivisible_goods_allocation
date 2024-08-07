# Indivisible Item Allocation

This repository contains implementations of various algorithms for the allocation of indivisible items among agents. It includes utilities to assist with common tasks in this domain.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Folder Structure](#folder-structure)
- [Usage](#usage)
- [Algorithms](#algorithms)
- [Utilities](#utilities)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project provides a collection of algorithms for solving the problem of allocating indivisible items to agents in a fair and efficient manner. It includes implementations of well-known algorithms and utility functions to facilitate experiments and comparisons.

## Features

- Implementation of various indivisible item allocation algorithms
- Utility functions for common operations
- Easy-to-follow code structure and examples

## Folder Structure

\```
├── algorithms
├── utils
├── allocation_algorithm_example.ipynb
├── README.md
└── LICENSE
\```

## Usage

You can find an example of usage in the notebook examples


## Algorithms

The `algorithms` folder contains the main allocation algorithms. Each algorithm is implemented in its own Python file. Here are some of the included algorithms:

- `barman.py`: algorithm described in https://arxiv.org/abs/1707.04731;
- `brute_force.py`: given an fair division problem instance compute all the EF1 and PO allocation;
- `envy_cycle.py`: implements the EnvyCycle Elimination algorithm, described in https://dl.acm.org/doi/10.1145/988772.988792;
- garg: algorithm described in https://arxiv.org/abs/2204.14229;
- `generalized_adjusted_winner.py`: implements a generalized adjusted winner procedure described in https://arxiv.org/abs/1807.10684;
- `minmaxenvy_trade.py`: implements a sequential mechanism, after adding one item in the pool, computes the minimax envy allocation that respect resource monotonicity;
- `mnw.py`: computes the allocation that maximize the Nash welfare, as it was proven working by https://dl.acm.org/doi/10.1145/3355902;
- `round_robin.py`: computes the round-robin procedure, that produced an EF1 allocaiton, explained in https://dl.acm.org/doi/10.1145/3355902. 

## Utilities

The `utils` folder contains utility functions that assist with common tasks such as data loading, preprocessing, and validation.

- `check.py`: class that implement a checker for various allocaiton axioms,
- `utils.py`: contains some useful functions as valuation profiles generator, ecc...
- `visualization.py`: function to visualize the allocation as a bipartite graph.
 
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to thank the authors and contributors of the algorithms included in this project.

