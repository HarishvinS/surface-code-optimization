## Overview

This project implements and analyzes fault-tolerant quantum circuits using PyMatching, Stim, and sinter. It demonstrates how quantum error correction can protect quantum algorithms from noise, which is crucial for practical quantum computing.

## Project Structure

### Files

1. **`main.py`** - Basic implementation with QNN circuits and surface code
2. **`enhanced_main.py`** - Comprehensive implementation with Shor's and Grover's algorithms
3. **`README.md`** - This documentation file

### Generated Files

- `qnn_fault_tolerance_results.png` - Basic analysis visualization
- `quantum_fault_tolerance_comprehensive_analysis.png` - Comprehensive analysis with multiple algorithms

## Features Implemented

### Stage 1: Setup and Installation
- Automatic verification of all required packages (Stim, PyMatching, sinter)
- Error handling for missing dependencies

### Stage 2: Learn the Basics 
- Implementation of quantum neural network circuits
- Basic quantum algorithm implementations (Shor's and Grover's)

### Stage 3: Select Quantum Algorithms 
- **Shor's Algorithm (Simplified)**: For integer factorization
- **Grover's Algorithm**: For database search
- **Quantum Neural Networks**: For machine learning applications

### Stage 4: Implement in Stim 
- Proper circuit construction using Stim's circuit language
- Noise modeling with depolarizing errors
- Surface code implementation (repetition code variant)

### Stage 5: Integrate PyMatching 
- Detector error model extraction
- Syndrome-based error decoding
- Minimum weight perfect matching for error correction

### Stage 6: Use sinter for Analysis 
- Statistical analysis of logical error rates
- Parallel simulation capabilities
- Performance benchmarking

### Stage 7: Analyze and Optimize 
- Comprehensive visualization of results
- Parameter optimization analysis
- Resource requirement estimation

## Key Results

### Performance Metrics

The project analyzes several key metrics:

1. **Logical Error Rate vs Physical Error Rate**: Shows how error correction improves with better codes
2. **Runtime Analysis**: Demonstrates computational costs of different code distances
3. **Error Rate Improvement**: Quantifies the benefit of higher distance codes
4. **Resource Requirements**: Estimates physical qubit needs

### Optimal Configurations Found

For the tested parameters:

- **Shor's Algorithm**: Best performance at distance 3 with 0.1% physical error rate
- **Grover's Algorithm**: Best performance at distance 3 with 0.1% physical error rate
- **Resource Cost**: 5 physical qubits required for distance-3 repetition code

## Technical Implementation

### Surface Code (Repetition Code)

The project implements a 1D repetition code as a simplified surface code:

- **Data Qubits**: Store the logical information
- **Syndrome Qubits**: Detect parity violations between adjacent data qubits
- **Multiple Rounds**: Syndrome extraction over multiple time steps
- **Logical Observable**: Parity of all data qubits

### Noise Model

Realistic noise is added using:

- **Depolarizing Noise**: Applied after each gate operation
- **Configurable Error Rates**: From 0.1% to 5% per gate
- **Two-Qubit Noise**: Additional noise for CNOT and CZ gates

### Error Decoding

PyMatching is used for:

- **Syndrome Processing**: Converting detector measurements to error syndromes
- **Minimum Weight Perfect Matching**: Finding most likely error patterns
- **Correction Application**: Determining logical corrections needed

## Usage

### Running the Basic Version

```bash
python main.py
```

This runs a simplified analysis with QNN circuits.

### Running the Enhanced Version

```bash
python enhanced_main.py
```

This runs comprehensive analysis with multiple quantum algorithms.

### Customizing Parameters

You can modify the following parameters in the code:

```python
code_distances = [3, 5, 7]  # Surface code distances
error_rates = [0.001, 0.01, 0.05]  # Physical error rates
shots = 1000  # Number of simulation runs
```

## Dependencies

Required Python packages:

- `stim` - Quantum circuit simulator
- `pymatching` - Error correction decoder
- `sinter` - Statistical analysis tools
- `numpy` - Numerical computations
- `matplotlib` - Visualization

Install with:

```bash
pip install stim pymatching sinter numpy matplotlib
```

## Results Interpretation

### Logical Error Rate

- **Lower is Better**: Indicates better error correction
- **Threshold Behavior**: Look for crossing points where higher distance codes become beneficial
- **Scaling**: Should decrease with code distance for error rates below threshold

### Resource Requirements

- **Qubit Overhead**: Repetition code requires `d + (d-1)` qubits for distance `d`
- **Time Overhead**: Multiple syndrome extraction rounds
- **Classical Processing**: PyMatching decoding computation

## Future Enhancements

Potential improvements to the project:

1. **2D Surface Codes**: Implement full topological surface codes
2. **More Algorithms**: Add Deutsch-Jozsa, VQE, QAOA
3. **Advanced Noise**: Include coherent errors, crosstalk
4. **Real Hardware**: Interface with actual quantum devices
5. **Optimization**: Automatic parameter tuning

## References

- [Stim Documentation](https://github.com/quantumlib/Stim)
- [PyMatching Documentation](https://github.com/oscarhiggott/PyMatching)
- [Surface Code Theory](https://arxiv.org/abs/1208.0928)
- [Quantum Error Correction](https://arxiv.org/abs/0904.2557)

## Contributing

To extend this project:

1. Fork the repository
2. Add new quantum algorithms in the algorithms section
3. Implement additional error correction codes
4. Enhance visualization and analysis
5. Submit pull requests with improvements
