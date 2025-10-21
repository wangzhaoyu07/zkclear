# Artifact for the paper "Privacy-preserving and Verifiable Causal Prescriptive Analytics"

This is a research artifact for the paper "Privacy-preserving and Verifiable Causal Prescriptive Analytics". The codebase demonstrates how to implement causal inference algorithms as zero-knowledge circuits using the Halo2 proving system.

## Repository Structure

The repository contains the following main components:

- `src/`: Core library code
  - `gadget/`: Zero-knowledge circuit implementations for causal algorithms
  - `scaffold/`: Framework for running circuits (keygen, prove, verify)
- `examples/`: Example applications demonstrating different algorithms
- `Cargo.toml`: Rust project configuration with Halo2 dependencies

## Dependencies

This project uses:
- **Halo2**: Zero-knowledge proof system
- **halo2-base**: Axiom's helper API for Halo2

## Installation

1. Install Rust (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Clone the repository:
```bash
git clone <repository-url>
cd zkclear
```

3. Build the project:
```bash
cargo build
```

## Usage

The project provides several primitive algorithms (see `src/gadget/` directory) and example applications (see `examples/` directory).

### Running Examples

Each example supports three modes:
- `keygen`: Generate proving and verification keys
- `prove`: Generate zero-knowledge proofs
- `verify`: Verify zero-knowledge proofs

#### Usage

```bash
# Run key generation
cargo run --example <example_name> -- --name <data_name> --num_attr <num_attributes> -k 8 keygen

# Generate proof
cargo run --example <example_name> -- --name <data_name> --num_attr <num_attributes> -k 8 prove

# Verify proof
cargo run --example <example_name> -- --name <data_name> --num_attr <num_attributes> -k 8 verify
```

## Data Files

The system expects data files in the `data/` directory. The exact format depends on the specific example being run.

<!-- ## License

MIT License -->

<!-- ## Citation

If you use this code in your research, please cite:

```
[Add citation information for the paper]
``` -->