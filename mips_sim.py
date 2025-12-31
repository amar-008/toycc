#!/usr/bin/env python3
"""
MIPS Interpreter for LiteCC Compiler

A simple MIPS assembly interpreter that executes the output of the LiteCC compiler.
Supports a subset of MIPS instructions sufficient for running compiled programs.

Author: Amarendra Mishra
"""
import sys
import re
from typing import Dict, List, Tuple


class MIPSInterpreter:
    """
    A simple MIPS assembly interpreter.

    Supports:
    - Arithmetic: add, addi, sub, mul, div, mflo, mfhi
    - Logic: and, andi, or, ori, xor, xori, nor
    - Shifts: sll, srl, sra, sllv, srlv, srav
    - Comparison: slt, slti, sltu, sltiu
    - Memory: lw, sw, lb, sb
    - Control: beq, bne, j, jal, jr
    - Immediate: li, la
    - Data movement: move
    - System calls: print_int, print_str, print_char, read_int
    """

    # Register name to index mapping
    REG_NAMES = {
        '$zero': 0, '$at': 1,
        '$v0': 2, '$v1': 3,
        '$a0': 4, '$a1': 5, '$a2': 6, '$a3': 7,
        '$t0': 8, '$t1': 9, '$t2': 10, '$t3': 11,
        '$t4': 12, '$t5': 13, '$t6': 14, '$t7': 15,
        '$s0': 16, '$s1': 17, '$s2': 18, '$s3': 19,
        '$s4': 20, '$s5': 21, '$s6': 22, '$s7': 23,
        '$t8': 24, '$t9': 25,
        '$k0': 26, '$k1': 27,
        '$gp': 28, '$sp': 29, '$fp': 30, '$ra': 31
    }

    def __init__(self, debug: bool = False):
        self.regs = [0] * 32
        self.memory: Dict[int, int] = {}
        self.data_strings: Dict[str, str] = {}
        self.labels: Dict[str, int] = {}
        self.pc = 0
        self.instructions: List[str] = []
        self.running = True
        self.lo = 0
        self.hi = 0
        self.debug = debug

    def get_reg(self, name: str) -> int:
        """Get the value of a register."""
        if name in self.REG_NAMES:
            idx = self.REG_NAMES[name]
            return 0 if idx == 0 else self.regs[idx]
        # Handle numeric register names like $0, $1, etc.
        if name.startswith('$') and name[1:].isdigit():
            idx = int(name[1:])
            return 0 if idx == 0 else self.regs[idx]
        raise RuntimeError(f"Unknown register: {name}")

    def set_reg(self, name: str, value: int) -> None:
        """Set the value of a register."""
        if name in self.REG_NAMES:
            idx = self.REG_NAMES[name]
            if idx != 0:  # $zero is always 0
                self.regs[idx] = value & 0xFFFFFFFF
            return
        if name.startswith('$') and name[1:].isdigit():
            idx = int(name[1:])
            if idx != 0:
                self.regs[idx] = value & 0xFFFFFFFF
            return
        raise RuntimeError(f"Unknown register: {name}")

    def signed(self, value: int) -> int:
        """Convert unsigned 32-bit value to signed."""
        if value & 0x80000000:
            return value - 0x100000000
        return value

    def read_word(self, addr: int) -> int:
        """Read a 32-bit word from memory (little-endian)."""
        word = 0
        for i in range(4):
            byte = self.memory.get(addr + i, 0)
            word |= (byte << (i * 8))
        return word

    def write_word(self, addr: int, value: int) -> None:
        """Write a 32-bit word to memory (little-endian)."""
        for i in range(4):
            self.memory[addr + i] = (value >> (i * 8)) & 0xFF

    def read_byte(self, addr: int) -> int:
        """Read a byte from memory."""
        return self.memory.get(addr, 0)

    def write_byte(self, addr: int, value: int) -> None:
        """Write a byte to memory."""
        self.memory[addr] = value & 0xFF

    def parse_offset_reg(self, operand: str) -> Tuple[int, str]:
        """Parse an offset(register) operand."""
        match = re.match(r'(-?\d+)\((\$\w+)\)', operand)
        if match:
            return int(match.group(1)), match.group(2)
        raise RuntimeError(f"Invalid offset(reg) format: {operand}")

    def parse_immediate(self, value: str) -> int:
        """Parse an immediate value (decimal or hex)."""
        value = value.strip()
        if value.startswith('0x') or value.startswith('0X'):
            return int(value, 16)
        if value.startswith('-0x') or value.startswith('-0X'):
            return -int(value[1:], 16)
        return int(value)

    def load_program(self, asm_code: str) -> None:
        """Load and parse a MIPS assembly program."""
        lines = asm_code.split('\n')
        in_data = False
        in_text = False
        current_string_label = None
        current_string = ""

        for line in lines:
            orig_line = line
            line = line.strip()

            # Section directives
            if line == '.data':
                in_data = True
                in_text = False
                if current_string_label:
                    self.data_strings[current_string_label] = current_string
                    current_string_label = None
                    current_string = ""
                continue
            elif line == '.text' or line.startswith('.globl'):
                in_data = False
                in_text = True
                if current_string_label:
                    self.data_strings[current_string_label] = current_string
                    current_string_label = None
                    current_string = ""
                continue

            if not line or line.startswith('#'):
                continue

            if in_data:
                # Parse data section
                match = re.match(r'(\w+):\s*\.asciiz\s*"(.*)', orig_line)
                if match:
                    if current_string_label:
                        self.data_strings[current_string_label] = current_string
                    current_string_label = match.group(1)
                    rest = match.group(2)
                    if rest.endswith('"'):
                        current_string = rest[:-1]
                        current_string = self._unescape(current_string)
                        self.data_strings[current_string_label] = current_string
                        current_string_label = None
                        current_string = ""
                    else:
                        current_string = rest + '\n'
                elif current_string_label:
                    if orig_line.rstrip().endswith('"'):
                        current_string += orig_line.rstrip()[:-1]
                        current_string = self._unescape(current_string)
                        self.data_strings[current_string_label] = current_string
                        current_string_label = None
                        current_string = ""
                    else:
                        current_string += orig_line + '\n'

            elif in_text:
                # Parse text section
                if ':' in line and not line.strip().startswith('#'):
                    label_match = re.match(r'(\w+):', line)
                    if label_match:
                        label = label_match.group(1)
                        self.labels[label] = len(self.instructions)
                        rest = line[line.index(':') + 1:].strip()
                        if rest and not rest.startswith('#'):
                            self.instructions.append(rest)
                    continue

                if line and not line.startswith('.'):
                    self.instructions.append(line)

        # Initialize stack pointer
        self.set_reg('$sp', 0x7FFFFFFC)

    def _unescape(self, s: str) -> str:
        """Unescape string escape sequences."""
        return (s.replace('\\n', '\n')
                 .replace('\\t', '\t')
                 .replace('\\r', '\r')
                 .replace('\\\\', '\\')
                 .replace('\\"', '"'))

    def run(self, max_instructions: int = 1000000) -> None:
        """Run the loaded program."""
        if 'main' not in self.labels:
            raise RuntimeError("No main function found")

        self.pc = self.labels['main']
        iterations = 0

        while self.running and iterations < max_instructions:
            if self.pc >= len(self.instructions):
                break

            inst = self.instructions[self.pc].strip()
            if not inst or inst.startswith('#'):
                self.pc += 1
                continue

            if self.debug:
                print(f"PC={self.pc}: {inst}")

            self.execute(inst)
            iterations += 1

        if iterations >= max_instructions:
            raise RuntimeError(f"Program exceeded maximum instruction count ({max_instructions})")

    def execute(self, inst: str) -> None:
        """Execute a single instruction."""
        # Remove comments
        if '#' in inst:
            inst = inst[:inst.index('#')]
        inst = inst.strip()

        if not inst:
            self.pc += 1
            return

        # Parse instruction
        parts = inst.split(None, 1)
        op = parts[0].lower()

        # Parse operands
        operands = []
        if len(parts) > 1:
            for part in parts[1].split(','):
                operands.append(part.strip())

        # Execute based on opcode
        if op == 'li':
            self.set_reg(operands[0], self.parse_immediate(operands[1]))

        elif op == 'la':
            label = operands[1]
            if label in self.data_strings:
                addr = 0x10000000 + hash(label) % 0x1000000
                self.set_reg(operands[0], addr)
                # Store string in memory
                string = self.data_strings[label]
                for i, char in enumerate(string):
                    self.memory[addr + i] = ord(char)
                self.memory[addr + len(string)] = 0  # Null terminator
            else:
                raise RuntimeError(f"Unknown label: {label}")

        elif op == 'move':
            self.set_reg(operands[0], self.get_reg(operands[1]))

        elif op == 'add':
            val1 = self.signed(self.get_reg(operands[1]))
            val2 = self.signed(self.get_reg(operands[2]))
            self.set_reg(operands[0], val1 + val2)

        elif op == 'addi':
            val = self.signed(self.get_reg(operands[1]))
            imm = self.parse_immediate(operands[2])
            self.set_reg(operands[0], val + imm)

        elif op == 'sub':
            val1 = self.signed(self.get_reg(operands[1]))
            val2 = self.signed(self.get_reg(operands[2]))
            self.set_reg(operands[0], val1 - val2)

        elif op == 'mul':
            val1 = self.signed(self.get_reg(operands[1]))
            val2 = self.signed(self.get_reg(operands[2]))
            self.set_reg(operands[0], val1 * val2)

        elif op == 'div':
            val1 = self.signed(self.get_reg(operands[0]))
            val2 = self.signed(self.get_reg(operands[1]))
            if val2 == 0:
                raise RuntimeError("Division by zero")
            self.lo = val1 // val2
            self.hi = val1 % val2

        elif op == 'mflo':
            self.set_reg(operands[0], self.lo)

        elif op == 'mfhi':
            self.set_reg(operands[0], self.hi)

        elif op == 'and':
            val1 = self.get_reg(operands[1])
            val2 = self.get_reg(operands[2])
            self.set_reg(operands[0], val1 & val2)

        elif op == 'andi':
            val = self.get_reg(operands[1])
            imm = self.parse_immediate(operands[2])
            self.set_reg(operands[0], val & imm)

        elif op == 'or':
            val1 = self.get_reg(operands[1])
            val2 = self.get_reg(operands[2])
            self.set_reg(operands[0], val1 | val2)

        elif op == 'ori':
            val = self.get_reg(operands[1])
            imm = self.parse_immediate(operands[2])
            self.set_reg(operands[0], val | imm)

        elif op == 'xor':
            val1 = self.get_reg(operands[1])
            val2 = self.get_reg(operands[2])
            self.set_reg(operands[0], val1 ^ val2)

        elif op == 'xori':
            val = self.get_reg(operands[1])
            imm = self.parse_immediate(operands[2])
            self.set_reg(operands[0], val ^ imm)

        elif op == 'nor':
            val1 = self.get_reg(operands[1])
            val2 = self.get_reg(operands[2])
            self.set_reg(operands[0], ~(val1 | val2))

        elif op == 'sll':
            val = self.get_reg(operands[1])
            shamt = self.parse_immediate(operands[2])
            self.set_reg(operands[0], val << shamt)

        elif op == 'srl':
            val = self.get_reg(operands[1])
            shamt = self.parse_immediate(operands[2])
            self.set_reg(operands[0], val >> shamt)

        elif op == 'sra':
            val = self.signed(self.get_reg(operands[1]))
            shamt = self.parse_immediate(operands[2])
            self.set_reg(operands[0], val >> shamt)

        elif op == 'sllv':
            val = self.get_reg(operands[1])
            shamt = self.get_reg(operands[2]) & 0x1F
            self.set_reg(operands[0], val << shamt)

        elif op == 'srlv':
            val = self.get_reg(operands[1])
            shamt = self.get_reg(operands[2]) & 0x1F
            self.set_reg(operands[0], val >> shamt)

        elif op == 'srav':
            val = self.signed(self.get_reg(operands[1]))
            shamt = self.get_reg(operands[2]) & 0x1F
            self.set_reg(operands[0], val >> shamt)

        elif op == 'slt':
            val1 = self.signed(self.get_reg(operands[1]))
            val2 = self.signed(self.get_reg(operands[2]))
            self.set_reg(operands[0], 1 if val1 < val2 else 0)

        elif op == 'slti':
            val = self.signed(self.get_reg(operands[1]))
            imm = self.parse_immediate(operands[2])
            self.set_reg(operands[0], 1 if val < imm else 0)

        elif op == 'sltu':
            val1 = self.get_reg(operands[1])
            val2 = self.get_reg(operands[2])
            self.set_reg(operands[0], 1 if val1 < val2 else 0)

        elif op == 'sltiu':
            val = self.get_reg(operands[1])
            imm = self.parse_immediate(operands[2]) & 0xFFFFFFFF
            self.set_reg(operands[0], 1 if val < imm else 0)

        elif op == 'lw':
            offset, reg = self.parse_offset_reg(operands[1])
            addr = self.get_reg(reg) + offset
            self.set_reg(operands[0], self.read_word(addr))

        elif op == 'sw':
            offset, reg = self.parse_offset_reg(operands[1])
            addr = self.get_reg(reg) + offset
            self.write_word(addr, self.get_reg(operands[0]))

        elif op == 'lb':
            offset, reg = self.parse_offset_reg(operands[1])
            addr = self.get_reg(reg) + offset
            val = self.read_byte(addr)
            # Sign extend
            if val & 0x80:
                val |= 0xFFFFFF00
            self.set_reg(operands[0], val)

        elif op == 'lbu':
            offset, reg = self.parse_offset_reg(operands[1])
            addr = self.get_reg(reg) + offset
            self.set_reg(operands[0], self.read_byte(addr))

        elif op == 'sb':
            offset, reg = self.parse_offset_reg(operands[1])
            addr = self.get_reg(reg) + offset
            self.write_byte(addr, self.get_reg(operands[0]))

        elif op == 'beq':
            val1 = self.get_reg(operands[0])
            val2 = self.get_reg(operands[1])
            if val1 == val2:
                self.pc = self.labels[operands[2]]
                return

        elif op == 'bne':
            val1 = self.get_reg(operands[0])
            val2 = self.get_reg(operands[1])
            if val1 != val2:
                self.pc = self.labels[operands[2]]
                return

        elif op == 'j':
            self.pc = self.labels[operands[0]]
            return

        elif op == 'jal':
            target = operands[0]
            # Built-in functions
            if target == 'print_int':
                val = self.signed(self.get_reg('$a0'))
                print(val, end='')
            elif target == 'print_str':
                addr = self.get_reg('$a0')
                chars = []
                while True:
                    byte = self.memory.get(addr, 0)
                    if byte == 0:
                        break
                    chars.append(chr(byte))
                    addr += 1
                print(''.join(chars), end='')
            elif target == 'print_char':
                val = self.get_reg('$a0') & 0xFF
                print(chr(val), end='')
            elif target == 'read_int':
                try:
                    val = int(input())
                    self.set_reg('$v0', val)
                except ValueError:
                    self.set_reg('$v0', 0)
            else:
                # User-defined function
                self.set_reg('$ra', self.pc + 1)
                if target in self.labels:
                    self.pc = self.labels[target]
                    return
                else:
                    raise RuntimeError(f"Unknown function: {target}")

        elif op == 'jr':
            if operands[0] == '$ra':
                target = self.get_reg('$ra')
                if target == 0:
                    self.running = False
                    return
                self.pc = target
                return
            else:
                self.pc = self.get_reg(operands[0])
                return

        elif op == 'nop':
            pass

        elif op == 'syscall':
            syscall_num = self.get_reg('$v0')
            if syscall_num == 1:  # print_int
                val = self.signed(self.get_reg('$a0'))
                print(val, end='')
            elif syscall_num == 4:  # print_str
                addr = self.get_reg('$a0')
                chars = []
                while True:
                    byte = self.memory.get(addr, 0)
                    if byte == 0:
                        break
                    chars.append(chr(byte))
                    addr += 1
                print(''.join(chars), end='')
            elif syscall_num == 5:  # read_int
                try:
                    val = int(input())
                    self.set_reg('$v0', val)
                except ValueError:
                    self.set_reg('$v0', 0)
            elif syscall_num == 10:  # exit
                self.running = False
                return
            elif syscall_num == 11:  # print_char
                val = self.get_reg('$a0') & 0xFF
                print(chr(val), end='')

        else:
            raise RuntimeError(f"Unknown instruction: {op}")

        self.pc += 1


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("MIPS Interpreter for LiteCC")
        print("")
        print("Usage: mips_sim.py <program.asm> [--debug]")
        print("")
        print("Options:")
        print("  --debug    Print each instruction as it executes")
        sys.exit(1)

    program_file = sys.argv[1]
    debug = '--debug' in sys.argv

    try:
        with open(program_file, 'r') as f:
            code = f.read()

        interpreter = MIPSInterpreter(debug=debug)
        interpreter.load_program(code)
        interpreter.run()

    except FileNotFoundError:
        print(f"Error: File not found: {program_file}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)


if __name__ == '__main__':
    main()
