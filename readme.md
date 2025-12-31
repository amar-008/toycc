# LiteCC - A Lightweight C to MIPS Compiler

[![CI](https://github.com/YOUR_USERNAME/litecc/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/litecc/actions/workflows/ci.yml)

A compiler for a subset of C that generates MIPS assembly code. Built from scratch to demonstrate compiler design concepts including lexical analysis, parsing, AST construction, and code generation.

## Features

### Language Support
- **Data Types**: `int`, `char`, `void`
- **Variables**: Local variables, function parameters, arrays
- **Operators**:
  - Arithmetic: `+`, `-`, `*`, `/`, `%`
  - Comparison: `<`, `>`, `<=`, `>=`, `==`, `!=`
  - Logical: `&&`, `||`, `!`
  - Bitwise: `&`, `|`, `^`, `~`, `<<`, `>>`
  - Assignment: `=`
  - Increment/Decrement: `++`, `--` (prefix and postfix)
- **Control Flow**: `if`/`else`, `while`, `for`, `break`, `continue`, `return`
- **Functions**: User-defined functions with parameters and return values, recursion
- **Comments**: Single-line (`//`) and multi-line (`/* */`)

### Compiler Features
- Detailed error messages with line and column numbers
- Debug modes: `--tokens` to view lexer output, `--ast` to view parsed AST
- Clean MIPS assembly output with proper stack management

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LiteCC Compiler                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Source Code (.c)                                              │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │  Lexer  │  Converts source into tokens                      │
│   └────┬────┘                                                   │
│        │ tokens                                                 │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │ Parser  │  Builds Abstract Syntax Tree                      │
│   └────┬────┘                                                   │
│        │ AST                                                    │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │ CodeGen │  Generates MIPS assembly                          │
│   └────┬────┘                                                   │
│        │                                                        │
│        ▼                                                        │
│   Assembly Code (.asm)                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Components

1. **Lexer** (`litecc.py:119-375`)
   - Tokenizes source code into a stream of tokens
   - Handles keywords, identifiers, literals, operators, and punctuation
   - Tracks line and column numbers for error reporting
   - Supports string escape sequences (`\n`, `\t`, etc.)

2. **Parser** (`litecc.py:544-1058`)
   - Recursive descent parser implementing C operator precedence
   - Builds a typed Abstract Syntax Tree (AST)
   - Handles all C expressions and statements

3. **Code Generator** (`litecc.py:1064-1559`)
   - Generates MIPS assembly from AST
   - Implements MIPS calling conventions
   - Stack-based variable allocation
   - Short-circuit evaluation for logical operators

4. **MIPS Interpreter** (`mips_sim.py`)
   - Executes generated MIPS assembly
   - Supports ~40 MIPS instructions
   - Built-in I/O functions

## Quick Start

### Requirements
- Python 3.9 or higher
- No external dependencies

### Basic Usage

```bash
# Compile a C file to MIPS assembly
python3 litecc.py input.c -o output.asm

# Run the generated assembly
python3 mips_sim.py output.asm

# Or use the Makefile
make run FILE=input.c
```

### Example Program

```c
// fizzbuzz.c
int main() {
    int i;
    for (i = 1; i <= 100; i++) {
        int by3 = i % 3;
        int by5 = i % 5;

        if (by3 == 0 && by5 == 0) {
            print_str("FizzBuzz\n");
        } else if (by3 == 0) {
            print_str("Fizz\n");
        } else if (by5 == 0) {
            print_str("Buzz\n");
        } else {
            print_int(i);
            print_str("\n");
        }
    }
    return 0;
}
```

```bash
make fizzbuzz
```

## Project Structure

```
litecc/
├── litecc.py          # Main compiler (lexer, parser, codegen)
├── mips_sim.py        # MIPS assembly interpreter
├── Makefile           # Build system
├── README.md          # This file
├── fizzbuzz.c         # Example program
├── tests/
│   ├── test_runner.py # Automated test runner
│   ├── test_arithmetic.c
│   ├── test_comparison.c
│   ├── test_logical.c
│   ├── test_bitwise.c
│   ├── test_loops.c
│   ├── test_break_continue.c
│   ├── test_arrays.c
│   ├── test_functions.c
│   ├── test_nested_loops.c
│   └── test_recursion.c
└── .github/
    └── workflows/
        └── ci.yml     # GitHub Actions CI pipeline
```

## Available Commands

```bash
make test              # Run all tests
make test-verbose      # Run tests with detailed output
make compile FILE=x.c  # Compile a C file
make run FILE=x.c      # Compile and run a C file
make tokens FILE=x.c   # Display lexer tokens
make ast FILE=x.c      # Display parsed AST (JSON)
make lint              # Run code linter
make clean             # Remove generated files
```

## Built-in Functions

| Function | Description |
|----------|-------------|
| `print_int(n)` | Print an integer |
| `print_str(s)` | Print a string |
| `print_char(c)` | Print a character |
| `read_int()` | Read an integer from stdin |

## Language Grammar (Simplified)

```
program     → function*
function    → type IDENT '(' params? ')' block
params      → param (',' param)*
param       → type IDENT ('[' ']')?
block       → '{' statement* '}'
statement   → declaration | if | while | for | return | break | continue | expr ';'
expression  → assignment
assignment  → logical_or ('=' assignment)?
logical_or  → logical_and ('||' logical_and)*
logical_and → equality ('&&' equality)*
equality    → comparison (('==' | '!=') comparison)*
comparison  → additive (('<' | '>' | '<=' | '>=') additive)*
additive    → multiplicative (('+' | '-') multiplicative)*
multiplicative → unary (('*' | '/' | '%') unary)*
unary       → ('!' | '-' | '~' | '++' | '--') unary | postfix
postfix     → primary ('++' | '--' | '[' expr ']' | '(' args? ')')*
primary     → NUMBER | STRING | CHAR | IDENT | '(' expression ')'
```

## Running Tests

```bash
# Run all tests
python3 tests/test_runner.py

# With verbose output
python3 tests/test_runner.py -v
```

Expected output:
```
============================================================
LiteCC Compiler Test Suite
============================================================

  PASS  arithmetic
  PASS  comparison
  PASS  logical
  PASS  bitwise
  PASS  loops
  PASS  break_continue
  PASS  arrays
  PASS  functions
  PASS  fizzbuzz
  PASS  nested_loops
  PASS  recursion

============================================================
Results: 11/11 passed
         All tests passed!
```

## Technical Details

### MIPS Code Generation

- Uses MIPS32 instruction set
- Stack grows downward (toward lower addresses)
- Frame pointer (`$fp`) based variable addressing
- Caller-saved registers: `$t0-$t9`
- Function arguments in `$a0-$a3`, additional on stack
- Return value in `$v0`

### Calling Convention

```
Stack Frame Layout:
┌─────────────────┐  Higher addresses
│  Arguments >4   │
├─────────────────┤
│  Return Addr    │  $fp + 4
├─────────────────┤
│  Saved $fp      │  $fp + 0
├─────────────────┤
│  Local Var 1    │  $fp - 4
│  Local Var 2    │  $fp - 8
│  ...            │
└─────────────────┘  Lower addresses ($sp)
```

## Limitations

- No floating-point numbers
- No pointers (except arrays)
- No structs, unions, or enums
- No preprocessor (#define, #include)
- No global variables
- Single compilation unit only

## Future Improvements

- [ ] Global variables
- [ ] Pointer arithmetic
- [ ] Struct/union support
- [ ] Optimization passes (constant folding, dead code elimination)
- [ ] x86-64 backend
- [ ] LLVM IR generation

## License

MIT License - feel free to use this project for learning and educational purposes.

## Author

Amarendra Mishra

---

*This project was created to demonstrate compiler construction concepts. It's suitable for educational purposes and as a portfolio project for software engineering internships.*
