#!/usr/bin/env python3
"""
Test Runner for LiteCC Compiler

Runs all test cases and reports results.
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunner:
    """Runs compiler test cases and validates output."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.root = Path(__file__).parent.parent
        self.compiler = self.root / "litecc.py"
        self.simulator = self.root / "mips_sim.py"
        self.tests_dir = self.root / "tests"

    def run_test(self, test_file: Path, expected_output: str, input_data: str = "") -> Tuple[bool, str]:
        """
        Run a single test case.

        Args:
            test_file: Path to the .c file
            expected_output: Expected program output
            input_data: Optional input to provide to the program

        Returns:
            Tuple of (passed, actual_output)
        """
        with tempfile.NamedTemporaryFile(suffix=".asm", delete=False) as f:
            asm_file = f.name

        try:
            # Compile
            result = subprocess.run(
                [sys.executable, str(self.compiler), str(test_file), "-o", asm_file],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return False, f"Compilation failed:\n{result.stderr}"

            # Run
            result = subprocess.run(
                [sys.executable, str(self.simulator), asm_file],
                capture_output=True,
                text=True,
                input=input_data,
                timeout=10
            )

            actual_output = result.stdout

            if actual_output.strip() == expected_output.strip():
                return True, actual_output
            else:
                return False, f"Expected:\n{expected_output}\n\nGot:\n{actual_output}"

        except subprocess.TimeoutExpired:
            return False, "Test timed out"
        except Exception as e:
            return False, f"Error: {e}"
        finally:
            if os.path.exists(asm_file):
                os.unlink(asm_file)

    def run_all_tests(self) -> bool:
        """Run all tests in the tests directory."""
        print("=" * 60)
        print("LiteCC Compiler Test Suite")
        print("=" * 60)
        print()

        # Discover tests
        test_cases = self.discover_tests()

        if not test_cases:
            print("No tests found!")
            return False

        # Run tests
        for test_name, test_file, expected, input_data in test_cases:
            self.run_single_test(test_name, test_file, expected, input_data)

        # Summary
        print()
        print("=" * 60)
        total = self.passed + self.failed
        print(f"Results: {self.passed}/{total} passed")

        if self.failed > 0:
            print(f"         {self.failed} FAILED")
            return False
        else:
            print("         All tests passed!")
            return True

    def discover_tests(self) -> List[Tuple[str, Path, str, str]]:
        """Discover test cases from the tests directory."""
        tests = []

        # Built-in test cases
        builtin_tests = [
            ("arithmetic", "test_arithmetic.c", "55\n-10\n600\n3\n2"),
            ("comparison", "test_comparison.c", "1\n0\n1\n1\n0\n1"),
            ("logical", "test_logical.c", "1\n0\n0\n1\n1\n0"),
            ("bitwise", "test_bitwise.c", "5\n7\n6\n-6\n20\n2"),
            ("loops", "test_loops.c", "55\n120\n10"),
            ("break_continue", "test_break_continue.c", "1\n2\n4\n5\n28"),
            ("arrays", "test_arrays.c", "1\n2\n3\n4\n5\n15"),
            ("functions", "test_functions.c", "120\n8\n55"),
            ("fizzbuzz", "fizzbuzz.c", self.get_fizzbuzz_output()),
            ("nested_loops", "test_nested_loops.c", "1\n2\n3\n2\n4\n6\n3\n6\n9"),
            ("recursion", "test_recursion.c", "55\n120"),
        ]

        for name, filename, expected in builtin_tests:
            test_path = self.tests_dir / filename
            if test_path.exists():
                tests.append((name, test_path, expected, ""))
            elif (self.root / filename).exists():
                tests.append((name, self.root / filename, expected, ""))

        return tests

    def get_fizzbuzz_output(self) -> str:
        """Generate expected FizzBuzz output."""
        lines = []
        for i in range(1, 101):
            if i % 15 == 0:
                lines.append("FizzBuzz")
            elif i % 3 == 0:
                lines.append("Fizz")
            elif i % 5 == 0:
                lines.append("Buzz")
            else:
                lines.append(str(i))
        return "\n".join(lines)

    def run_single_test(self, name: str, test_file: Path, expected: str, input_data: str):
        """Run a single test and print result."""
        passed, output = self.run_test(test_file, expected, input_data)

        if passed:
            self.passed += 1
            print(f"  PASS  {name}")
        else:
            self.failed += 1
            print(f"  FAIL  {name}")
            if self.verbose:
                print(f"        {output}")


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    runner = TestRunner(verbose=verbose)

    if runner.run_all_tests():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
