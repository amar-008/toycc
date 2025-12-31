# LiteCC Compiler Makefile
# A lightweight C to MIPS compiler

PYTHON := python3
COMPILER := litecc.py
SIMULATOR := mips_sim.py
TEST_RUNNER := tests/test_runner.py

# Source files
SOURCES := $(wildcard *.c) $(wildcard tests/*.c)
ASM_FILES := $(SOURCES:.c=.asm)

.PHONY: all test clean help compile run lint format

# Default target
all: help

# Run all tests
test:
	@echo "Running test suite..."
	@$(PYTHON) $(TEST_RUNNER)

# Run tests with verbose output
test-verbose:
	@echo "Running test suite (verbose)..."
	@$(PYTHON) $(TEST_RUNNER) -v

# Compile a single file (usage: make compile FILE=example.c)
compile:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make compile FILE=<source.c>"; \
		exit 1; \
	fi
	$(PYTHON) $(COMPILER) $(FILE) -o $(FILE:.c=.asm)

# Compile and run a single file (usage: make run FILE=example.c)
run:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make run FILE=<source.c>"; \
		exit 1; \
	fi
	$(PYTHON) $(COMPILER) $(FILE) -o $(FILE:.c=.asm)
	$(PYTHON) $(SIMULATOR) $(FILE:.c=.asm)

# Run the fizzbuzz example
fizzbuzz:
	$(PYTHON) $(COMPILER) fizzbuzz.c -o fizzbuzz.asm
	$(PYTHON) $(SIMULATOR) fizzbuzz.asm

# Show tokens for a file (usage: make tokens FILE=example.c)
tokens:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make tokens FILE=<source.c>"; \
		exit 1; \
	fi
	$(PYTHON) $(COMPILER) $(FILE) --tokens

# Show AST for a file (usage: make ast FILE=example.c)
ast:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make ast FILE=<source.c>"; \
		exit 1; \
	fi
	$(PYTHON) $(COMPILER) $(FILE) --ast

# Run Python linter
lint:
	@echo "Running linter..."
	@$(PYTHON) -m py_compile $(COMPILER) $(SIMULATOR) $(TEST_RUNNER) 2>&1 || true
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check $(COMPILER) $(SIMULATOR); \
	elif command -v flake8 >/dev/null 2>&1; then \
		flake8 $(COMPILER) $(SIMULATOR) --max-line-length=120; \
	else \
		echo "No linter found (install ruff or flake8)"; \
	fi

# Type checking
typecheck:
	@echo "Running type checker..."
	@if command -v mypy >/dev/null 2>&1; then \
		mypy $(COMPILER) $(SIMULATOR) --ignore-missing-imports; \
	else \
		echo "mypy not found (install with: pip install mypy)"; \
	fi

# Clean generated files
clean:
	rm -f *.asm
	rm -f tests/*.asm
	rm -f __pycache__/*.pyc
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf .mypy_cache
	rm -rf .ruff_cache

# Install development dependencies
dev-setup:
	pip install mypy ruff

# Show help
help:
	@echo "LiteCC Compiler - Build System"
	@echo ""
	@echo "Usage:"
	@echo "  make test              Run the test suite"
	@echo "  make test-verbose      Run tests with verbose output"
	@echo "  make compile FILE=x.c  Compile a C file to MIPS assembly"
	@echo "  make run FILE=x.c      Compile and run a C file"
	@echo "  make fizzbuzz          Run the FizzBuzz example"
	@echo "  make tokens FILE=x.c   Show lexer tokens"
	@echo "  make ast FILE=x.c      Show parsed AST"
	@echo "  make lint              Run Python linter"
	@echo "  make typecheck         Run type checker"
	@echo "  make clean             Remove generated files"
	@echo "  make dev-setup         Install development tools"
	@echo "  make help              Show this help message"
