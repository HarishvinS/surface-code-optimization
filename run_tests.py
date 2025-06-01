#!/usr/bin/env python3
"""
Test runner for the Quantum Fault Tolerance Project.
This script allows you to run specific tests and analyses.
"""

import sys
import argparse
from shor_and_grover_main import *

def test_installation():
    """Test if all packages are properly installed."""
    print("Testing installation...")
    return verify_installation()

def test_surface_code_basic():
    """Test basic surface code functionality."""
    print("Testing surface code implementation...")
    try:
        # Create a simple surface code
        circuit = create_surface_code_repetition(3)
        print(f"✓ Surface code created with {circuit.num_qubits} qubits")
        
        # Test detector error model
        dem = circuit.detector_error_model(decompose_errors=True)
        print(f"✓ Detector error model created with {dem.num_detectors} detectors")
        
        # Test PyMatching integration
        matcher = setup_matching_graph(circuit)
        print("✓ PyMatching integration successful")
        
        return True
    except Exception as e:
        print(f"✗ Surface code test failed: {e}")
        return False

def test_algorithms():
    """Test quantum algorithm implementations."""
    print("Testing quantum algorithms...")
    
    algorithms = {
        "Shor's Algorithm": create_shors_algorithm_circuit,
        "Grover's Algorithm": create_grovers_algorithm_circuit,
    }
    
    for name, func in algorithms.items():
        try:
            circuit = func()
            print(f"✓ {name} circuit created with {circuit.num_qubits} qubits")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            return False
    
    return True

def test_noise_model():
    """Test noise model implementation."""
    print("Testing noise model...")
    try:
        # Create a simple circuit
        circuit = create_grovers_algorithm_circuit()
        
        # Add noise
        noisy_circuit = add_noise_to_circuit(circuit, 0.01)
        print(f"✓ Noise added to circuit")
        
        return True
    except Exception as e:
        print(f"✗ Noise model test failed: {e}")
        return False

def run_quick_analysis():
    """Run a quick analysis with minimal parameters."""
    print("Running quick analysis...")
    
    try:
        # Test with minimal parameters
        algorithm_circuit = create_grovers_algorithm_circuit()
        results = run_algorithm_analysis(
            "Grover's (Quick Test)", 
            algorithm_circuit, 
            code_distances=[3], 
            error_rates=[0.01]
        )
        
        if results and 3 in results and 0.01 in results[3]:
            error_rate = results[3][0.01].get("logical_error_rate", "N/A")
            print(f"✓ Quick analysis completed. Logical error rate: {error_rate}")
            return True
        else:
            print("✗ Quick analysis failed - no results")
            return False
            
    except Exception as e:
        print(f"✗ Quick analysis failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests."""
    print("=" * 60)
    print("COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Installation", test_installation),
        ("Surface Code", test_surface_code_basic),
        ("Algorithms", test_algorithms),
        ("Noise Model", test_noise_model),
        ("Quick Analysis", run_quick_analysis),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name} Test:")
        print("-" * 30)
        results[test_name] = test_func()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:<20}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

def main():
    parser = argparse.ArgumentParser(description="Quantum Fault Tolerance Project Test Runner")
    parser.add_argument("--test", choices=[
        "installation", "surface-code", "algorithms", "noise", "quick", "all"
    ], default="all", help="Which test to run")
    
    parser.add_argument("--run-full", action="store_true", 
                       help="Run the full comprehensive analysis")
    
    args = parser.parse_args()
    
    if args.run_full:
        print("Running full comprehensive analysis...")
        run_comprehensive_quantum_fault_tolerance_project()
        return
    
    # Run specific tests
    test_map = {
        "installation": test_installation,
        "surface-code": test_surface_code_basic,
        "algorithms": test_algorithms,
        "noise": test_noise_model,
        "quick": run_quick_analysis,
        "all": run_comprehensive_test,
    }
    
    test_func = test_map[args.test]
    success = test_func()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()