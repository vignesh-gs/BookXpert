#!/usr/bin/env python3
"""Runs pytest and CLI smoke tests. Usage: python run_tests.py"""

import subprocess
import sys
import os
from pathlib import Path


def run_pytest():
    """Run pytest."""
    print("=" * 60)
    print("Running pytest test suite...")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=Path(__file__).parent
    )
    return result.returncode == 0


def run_smoke_tests():
    """Smoke tests via CLI."""
    print("\n" + "=" * 60)
    print("Running smoke tests...")
    print("=" * 60)
    
    project_dir = Path(__file__).parent
    
    test_cases = [
        ("Geetha B.S", "Geetha"),
        ("Vignesh G.S", "Vignesh"),
        ("Krishna R", "Krishna"),
    ]
    
    all_passed = True
    
    for query, expected in test_cases:
        print(f"\nTest: Query='{query}', expecting '{expected}' in results...")
        
        result = subprocess.run(
            [sys.executable, "-m", "src.cli", "--name", query, "--top_k", "3"],
            cwd=project_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"  FAIL: CLI returned exit code {result.returncode}")
            print(f"  stderr: {result.stderr}")
            all_passed = False
            continue
        
        if expected.lower() in result.stdout.lower():
            print(f"  PASS: Found '{expected}' in output")
        else:
            print(f"  FAIL: '{expected}' not found in output")
            print(f"  Output: {result.stdout[:500]}")
            all_passed = False
    
    return all_passed


def main():
    """Pytest + smoke tests."""
    os.chdir(Path(__file__).parent)
    
    pytest_passed = run_pytest()
    smoke_passed = run_smoke_tests()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"  Pytest:      {'PASSED' if pytest_passed else 'FAILED'}")
    print(f"  Smoke tests: {'PASSED' if smoke_passed else 'FAILED'}")
    print("=" * 60)
    
    if pytest_passed and smoke_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
