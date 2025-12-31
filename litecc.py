#!/usr/bin/env python3
"""
LiteCC - A Lightweight C to MIPS Compiler
A compiler for a subset of C, demonstrating lexical analysis, parsing, and code generation.

Author: Amarendra Mishra
"""
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
from enum import Enum, auto


# =============================================================================
# TOKEN DEFINITIONS
# =============================================================================

class TokenType(Enum):
    """Enumeration of all token types recognized by the lexer."""
    # Literals
    NUMBER = auto()
    STRING = auto()
    CHAR = auto()

    # Identifiers and keywords
    IDENT = auto()
    KEYWORD = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    ASSIGN = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    INC = auto()
    DEC = auto()
    AND = auto()      # &&
    OR = auto()       # ||
    NOT = auto()      # !
    BIT_AND = auto()  # &
    BIT_OR = auto()   # |
    BIT_XOR = auto()  # ^
    BIT_NOT = auto()  # ~
    SHL = auto()      # <<
    SHR = auto()      # >>

    # Punctuation
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMICOLON = auto()
    COMMA = auto()

    # End of file
    EOF = auto()


@dataclass
class Token:
    """Represents a single token from the source code."""
    type: TokenType
    value: Any
    line: int
    column: int

    def __repr__(self):
        return f"Token({self.type.name}, {repr(self.value)}, line={self.line})"


# =============================================================================
# ERROR HANDLING
# =============================================================================

class CompilerError(Exception):
    """Base class for compiler errors with source location information."""
    def __init__(self, message: str, line: int = 0, column: int = 0):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(self.format_error())

    def format_error(self) -> str:
        if self.line > 0:
            return f"Error at line {self.line}, column {self.column}: {self.message}"
        return f"Error: {self.message}"


class LexerError(CompilerError):
    """Error during lexical analysis."""
    pass


class ParseError(CompilerError):
    """Error during parsing."""
    pass


class SemanticError(CompilerError):
    """Error during semantic analysis."""
    pass


class CodeGenError(CompilerError):
    """Error during code generation."""
    pass


# =============================================================================
# LEXER
# =============================================================================

class Lexer:
    """
    Lexical analyzer that converts source code into tokens.

    Supports:
    - Single and multi-line comments
    - Integer and character literals
    - String literals with escape sequences
    - All C operators including logical and bitwise
    - Keywords: int, char, void, if, else, while, for, return, break, continue
    """

    KEYWORDS = {'int', 'char', 'void', 'if', 'else', 'while', 'for',
                'return', 'break', 'continue', 'do', 'switch', 'case', 'default'}

    def __init__(self, source: str, filename: str = "<stdin>"):
        self.source = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

    def error(self, message: str) -> None:
        raise LexerError(message, self.line, self.column)

    def peek(self, offset: int = 0) -> Optional[str]:
        """Look at character at current position + offset without consuming."""
        idx = self.pos + offset
        return self.source[idx] if idx < len(self.source) else None

    def advance(self) -> Optional[str]:
        """Consume and return the current character."""
        if self.pos >= len(self.source):
            return None
        char = self.source[self.pos]
        self.pos += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char

    def skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.peek() and self.peek().isspace():
            self.advance()

    def skip_comment(self) -> bool:
        """Skip single-line (//) and multi-line (/* */) comments."""
        if self.peek() == '/' and self.peek(1) == '/':
            # Single-line comment
            while self.peek() and self.peek() != '\n':
                self.advance()
            return True

        if self.peek() == '/' and self.peek(1) == '*':
            # Multi-line comment
            start_line = self.line
            self.advance()  # /
            self.advance()  # *
            while True:
                if self.peek() is None:
                    self.error(f"Unterminated comment starting at line {start_line}")
                if self.peek() == '*' and self.peek(1) == '/':
                    self.advance()  # *
                    self.advance()  # /
                    break
                self.advance()
            return True

        return False

    def read_number(self) -> Token:
        """Read an integer literal."""
        start_col = self.column
        num_str = ''

        # Handle hex literals
        if self.peek() == '0' and self.peek(1) in ('x', 'X'):
            num_str = self.advance() + self.advance()  # 0x
            while self.peek() and (self.peek().isdigit() or self.peek().lower() in 'abcdef'):
                num_str += self.advance()
            return Token(TokenType.NUMBER, int(num_str, 16), self.line, start_col)

        while self.peek() and self.peek().isdigit():
            num_str += self.advance()

        return Token(TokenType.NUMBER, int(num_str), self.line, start_col)

    def read_string(self) -> Token:
        """Read a string literal with escape sequence support."""
        start_line = self.line
        start_col = self.column
        self.advance()  # Opening quote

        result = ''
        while self.peek() and self.peek() != '"':
            if self.peek() == '\n':
                self.error("Unterminated string literal")
            if self.peek() == '\\':
                self.advance()
                escape = self.advance()
                if escape == 'n':
                    result += '\n'
                elif escape == 't':
                    result += '\t'
                elif escape == 'r':
                    result += '\r'
                elif escape == '\\':
                    result += '\\'
                elif escape == '"':
                    result += '"'
                elif escape == '0':
                    result += '\0'
                else:
                    result += escape
            else:
                result += self.advance()

        if self.peek() != '"':
            self.error(f"Unterminated string starting at line {start_line}")

        self.advance()  # Closing quote
        return Token(TokenType.STRING, result, start_line, start_col)

    def read_char(self) -> Token:
        """Read a character literal."""
        start_col = self.column
        self.advance()  # Opening quote

        if self.peek() == '\\':
            self.advance()
            escape = self.advance()
            if escape == 'n':
                value = ord('\n')
            elif escape == 't':
                value = ord('\t')
            elif escape == 'r':
                value = ord('\r')
            elif escape == '\\':
                value = ord('\\')
            elif escape == "'":
                value = ord("'")
            elif escape == '0':
                value = 0
            else:
                value = ord(escape)
        else:
            value = ord(self.advance())

        if self.peek() != "'":
            self.error("Unterminated character literal")
        self.advance()  # Closing quote

        return Token(TokenType.CHAR, value, self.line, start_col)

    def read_identifier(self) -> Token:
        """Read an identifier or keyword."""
        start_col = self.column
        ident = ''

        while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
            ident += self.advance()

        if ident in self.KEYWORDS:
            return Token(TokenType.KEYWORD, ident, self.line, start_col)
        return Token(TokenType.IDENT, ident, self.line, start_col)

    def read_operator(self) -> Token:
        """Read an operator or punctuation token."""
        start_col = self.column
        char = self.peek()

        # Two-character operators
        two_char = char + (self.peek(1) or '')

        two_char_ops = {
            '==': TokenType.EQ,
            '!=': TokenType.NE,
            '<=': TokenType.LE,
            '>=': TokenType.GE,
            '++': TokenType.INC,
            '--': TokenType.DEC,
            '&&': TokenType.AND,
            '||': TokenType.OR,
            '<<': TokenType.SHL,
            '>>': TokenType.SHR,
        }

        if two_char in two_char_ops:
            self.advance()
            self.advance()
            return Token(two_char_ops[two_char], two_char, self.line, start_col)

        # Single-character operators
        single_char_ops = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.STAR,
            '/': TokenType.SLASH,
            '%': TokenType.PERCENT,
            '=': TokenType.ASSIGN,
            '<': TokenType.LT,
            '>': TokenType.GT,
            '!': TokenType.NOT,
            '&': TokenType.BIT_AND,
            '|': TokenType.BIT_OR,
            '^': TokenType.BIT_XOR,
            '~': TokenType.BIT_NOT,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            ';': TokenType.SEMICOLON,
            ',': TokenType.COMMA,
        }

        if char in single_char_ops:
            self.advance()
            return Token(single_char_ops[char], char, self.line, start_col)

        self.error(f"Unexpected character: '{char}'")

    def tokenize(self) -> List[Token]:
        """Convert source code into a list of tokens."""
        while self.pos < len(self.source):
            self.skip_whitespace()

            if self.pos >= len(self.source):
                break

            if self.skip_comment():
                continue

            char = self.peek()

            if char.isdigit():
                self.tokens.append(self.read_number())
            elif char == '"':
                self.tokens.append(self.read_string())
            elif char == "'":
                self.tokens.append(self.read_char())
            elif char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
            else:
                self.tokens.append(self.read_operator())

        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens


# =============================================================================
# AST NODE DEFINITIONS
# =============================================================================

@dataclass
class Program:
    """Root node containing all function definitions."""
    functions: List['Function']
    line: int = 0


@dataclass
class Function:
    """Function definition with return type, name, parameters, and body."""
    return_type: str
    name: str
    params: List['Parameter']
    body: List['Statement']
    line: int = 0


@dataclass
class Parameter:
    """Function parameter with type and name."""
    param_type: str
    name: str
    is_array: bool = False
    line: int = 0


@dataclass
class Declaration:
    """Variable declaration with optional initialization."""
    var_type: str
    name: str
    init: Optional['Expression'] = None
    array_size: Optional[int] = None
    line: int = 0


@dataclass
class ExpressionStmt:
    """Expression used as a statement."""
    expr: 'Expression'
    line: int = 0


@dataclass
class IfStmt:
    """If-else statement."""
    condition: 'Expression'
    then_branch: List['Statement']
    else_branch: Optional[List['Statement']] = None
    line: int = 0


@dataclass
class WhileStmt:
    """While loop."""
    condition: 'Expression'
    body: List['Statement']
    line: int = 0


@dataclass
class ForStmt:
    """For loop."""
    init: Optional[Union['Statement', 'Expression']] = None
    condition: Optional['Expression'] = None
    update: Optional['Expression'] = None
    body: Optional[List['Statement']] = None
    line: int = 0


@dataclass
class ReturnStmt:
    """Return statement."""
    value: Optional['Expression'] = None
    line: int = 0


@dataclass
class BreakStmt:
    """Break statement."""
    line: int = 0


@dataclass
class ContinueStmt:
    """Continue statement."""
    line: int = 0


@dataclass
class NumberExpr:
    """Integer literal."""
    value: int
    line: int = 0


@dataclass
class StringExpr:
    """String literal."""
    value: str
    line: int = 0


@dataclass
class VarExpr:
    """Variable reference."""
    name: str
    line: int = 0


@dataclass
class ArrayAccessExpr:
    """Array element access."""
    array: str
    index: 'Expression'
    line: int = 0


@dataclass
class AssignExpr:
    """Assignment expression."""
    target: 'Expression'
    value: 'Expression'
    line: int = 0


@dataclass
class BinaryExpr:
    """Binary operation."""
    op: str
    left: 'Expression'
    right: 'Expression'
    line: int = 0


@dataclass
class UnaryExpr:
    """Unary operation (prefix)."""
    op: str
    operand: 'Expression'
    line: int = 0


@dataclass
class PostfixExpr:
    """Postfix operation (++ or --)."""
    op: str
    operand: 'Expression'
    line: int = 0


@dataclass
class CallExpr:
    """Function call."""
    name: str
    args: List['Expression']
    line: int = 0


# Type aliases for clarity
Statement = Union[Declaration, ExpressionStmt, IfStmt, WhileStmt, ForStmt, ReturnStmt, BreakStmt, ContinueStmt]
Expression = Union[NumberExpr, StringExpr, VarExpr, ArrayAccessExpr, AssignExpr, BinaryExpr, UnaryExpr, PostfixExpr, CallExpr]


# =============================================================================
# PARSER
# =============================================================================

class Parser:
    """
    Recursive descent parser for ToyC.

    Grammar (simplified):
        program     -> function*
        function    -> type IDENT '(' params? ')' block
        params      -> param (',' param)*
        param       -> type IDENT ('[' ']')?
        block       -> '{' statement* '}'
        statement   -> declaration | if | while | for | return | break | continue | expr_stmt
        expression  -> assignment
        assignment  -> logical_or ('=' assignment)?
        logical_or  -> logical_and ('||' logical_and)*
        logical_and -> bitwise_or ('&&' bitwise_or)*
        bitwise_or  -> bitwise_xor ('|' bitwise_xor)*
        bitwise_xor -> bitwise_and ('^' bitwise_and)*
        bitwise_and -> equality ('&' equality)*
        equality    -> comparison (('==' | '!=') comparison)*
        comparison  -> shift (('<' | '>' | '<=' | '>=') shift)*
        shift       -> additive (('<<' | '>>') additive)*
        additive    -> multiplicative (('+' | '-') multiplicative)*
        multiplicative -> unary (('*' | '/' | '%') unary)*
        unary       -> ('!' | '-' | '~' | '++' | '--') unary | postfix
        postfix     -> primary ('++' | '--' | '[' expr ']' | '(' args? ')')*
        primary     -> NUMBER | STRING | IDENT | '(' expression ')'
    """

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def error(self, message: str) -> None:
        token = self.current()
        raise ParseError(message, token.line, token.column)

    def current(self) -> Token:
        """Get current token."""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else self.tokens[-1]

    def peek(self, offset: int = 0) -> Token:
        """Look ahead at token."""
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else self.tokens[-1]

    def check(self, token_type: TokenType, value: Any = None) -> bool:
        """Check if current token matches type and optionally value."""
        token = self.current()
        if token.type != token_type:
            return False
        if value is not None and token.value != value:
            return False
        return True

    def match(self, token_type: TokenType, value: Any = None) -> bool:
        """Match and consume token if it matches."""
        if self.check(token_type, value):
            self.pos += 1
            return True
        return False

    def consume(self, token_type: TokenType, value: Any = None, message: str = None) -> Token:
        """Consume token, raising error if it doesn't match."""
        token = self.current()
        if not self.check(token_type, value):
            expected = f"'{value}'" if value else token_type.name
            msg = message or f"Expected {expected}, got '{token.value}'"
            self.error(msg)
        self.pos += 1
        return token

    def parse(self) -> Program:
        """Parse the entire program."""
        functions = []
        while not self.check(TokenType.EOF):
            functions.append(self.parse_function())
        return Program(functions=functions, line=1)

    def parse_function(self) -> Function:
        """Parse a function definition."""
        line = self.current().line

        # Return type
        return_type = self.consume(TokenType.KEYWORD).value
        if return_type not in ('int', 'void', 'char'):
            self.error(f"Invalid return type: {return_type}")

        # Function name
        name = self.consume(TokenType.IDENT).value

        # Parameters
        self.consume(TokenType.LPAREN)
        params = []

        if not self.check(TokenType.RPAREN):
            params.append(self.parse_parameter())
            while self.match(TokenType.COMMA):
                params.append(self.parse_parameter())

        self.consume(TokenType.RPAREN)

        # Body
        body = self.parse_block()

        return Function(return_type=return_type, name=name, params=params, body=body, line=line)

    def parse_parameter(self) -> Parameter:
        """Parse a function parameter."""
        line = self.current().line
        param_type = self.consume(TokenType.KEYWORD).value
        name = self.consume(TokenType.IDENT).value

        is_array = False
        if self.match(TokenType.LBRACKET):
            self.consume(TokenType.RBRACKET)
            is_array = True

        return Parameter(param_type=param_type, name=name, is_array=is_array, line=line)

    def parse_block(self) -> List[Statement]:
        """Parse a block of statements."""
        self.consume(TokenType.LBRACE)
        statements = []

        while not self.check(TokenType.RBRACE) and not self.check(TokenType.EOF):
            statements.append(self.parse_statement())

        self.consume(TokenType.RBRACE)
        return statements

    def parse_statement(self) -> Statement:
        """Parse a single statement."""
        if self.check(TokenType.KEYWORD):
            keyword = self.current().value

            if keyword in ('int', 'char', 'void'):
                return self.parse_declaration()
            elif keyword == 'if':
                return self.parse_if()
            elif keyword == 'while':
                return self.parse_while()
            elif keyword == 'for':
                return self.parse_for()
            elif keyword == 'return':
                return self.parse_return()
            elif keyword == 'break':
                return self.parse_break()
            elif keyword == 'continue':
                return self.parse_continue()

        return self.parse_expression_stmt()

    def parse_declaration(self) -> Declaration:
        """Parse a variable declaration."""
        line = self.current().line
        var_type = self.consume(TokenType.KEYWORD).value
        name = self.consume(TokenType.IDENT).value

        # Check for array declaration
        array_size = None
        if self.match(TokenType.LBRACKET):
            size_token = self.consume(TokenType.NUMBER)
            array_size = size_token.value
            self.consume(TokenType.RBRACKET)

        # Check for initialization
        init = None
        if self.match(TokenType.ASSIGN):
            init = self.parse_expression()

        self.consume(TokenType.SEMICOLON)
        return Declaration(var_type=var_type, name=name, init=init, array_size=array_size, line=line)

    def parse_if(self) -> IfStmt:
        """Parse an if statement."""
        line = self.current().line
        self.consume(TokenType.KEYWORD, 'if')
        self.consume(TokenType.LPAREN)
        condition = self.parse_expression()
        self.consume(TokenType.RPAREN)

        # Then branch
        if self.check(TokenType.LBRACE):
            then_branch = self.parse_block()
        else:
            then_branch = [self.parse_statement()]

        # Else branch
        else_branch = None
        if self.match(TokenType.KEYWORD, 'else'):
            if self.check(TokenType.LBRACE):
                else_branch = self.parse_block()
            else:
                else_branch = [self.parse_statement()]

        return IfStmt(condition=condition, then_branch=then_branch, else_branch=else_branch, line=line)

    def parse_while(self) -> WhileStmt:
        """Parse a while loop."""
        line = self.current().line
        self.consume(TokenType.KEYWORD, 'while')
        self.consume(TokenType.LPAREN)
        condition = self.parse_expression()
        self.consume(TokenType.RPAREN)

        if self.check(TokenType.LBRACE):
            body = self.parse_block()
        else:
            body = [self.parse_statement()]

        return WhileStmt(condition=condition, body=body, line=line)

    def parse_for(self) -> ForStmt:
        """Parse a for loop."""
        line = self.current().line
        self.consume(TokenType.KEYWORD, 'for')
        self.consume(TokenType.LPAREN)

        # Init
        init = None
        if not self.check(TokenType.SEMICOLON):
            if self.check(TokenType.KEYWORD) and self.current().value in ('int', 'char'):
                init = self.parse_declaration()
            else:
                init = self.parse_expression()
                self.consume(TokenType.SEMICOLON)
        else:
            self.consume(TokenType.SEMICOLON)

        # Condition
        condition = None
        if not self.check(TokenType.SEMICOLON):
            condition = self.parse_expression()
        self.consume(TokenType.SEMICOLON)

        # Update
        update = None
        if not self.check(TokenType.RPAREN):
            update = self.parse_expression()
        self.consume(TokenType.RPAREN)

        # Body
        if self.check(TokenType.LBRACE):
            body = self.parse_block()
        else:
            body = [self.parse_statement()]

        return ForStmt(init=init, condition=condition, update=update, body=body, line=line)

    def parse_return(self) -> ReturnStmt:
        """Parse a return statement."""
        line = self.current().line
        self.consume(TokenType.KEYWORD, 'return')

        value = None
        if not self.check(TokenType.SEMICOLON):
            value = self.parse_expression()

        self.consume(TokenType.SEMICOLON)
        return ReturnStmt(value=value, line=line)

    def parse_break(self) -> BreakStmt:
        """Parse a break statement."""
        line = self.current().line
        self.consume(TokenType.KEYWORD, 'break')
        self.consume(TokenType.SEMICOLON)
        return BreakStmt(line=line)

    def parse_continue(self) -> ContinueStmt:
        """Parse a continue statement."""
        line = self.current().line
        self.consume(TokenType.KEYWORD, 'continue')
        self.consume(TokenType.SEMICOLON)
        return ContinueStmt(line=line)

    def parse_expression_stmt(self) -> ExpressionStmt:
        """Parse an expression statement."""
        line = self.current().line
        expr = self.parse_expression()
        self.consume(TokenType.SEMICOLON)
        return ExpressionStmt(expr=expr, line=line)

    def parse_expression(self) -> Expression:
        """Parse an expression (entry point for expression parsing)."""
        return self.parse_assignment()

    def parse_assignment(self) -> Expression:
        """Parse assignment expression."""
        expr = self.parse_logical_or()

        if self.match(TokenType.ASSIGN):
            value = self.parse_assignment()
            return AssignExpr(target=expr, value=value, line=expr.line)

        return expr

    def parse_logical_or(self) -> Expression:
        """Parse logical OR expression."""
        left = self.parse_logical_and()

        while self.match(TokenType.OR):
            right = self.parse_logical_and()
            left = BinaryExpr(op='||', left=left, right=right, line=left.line)

        return left

    def parse_logical_and(self) -> Expression:
        """Parse logical AND expression."""
        left = self.parse_bitwise_or()

        while self.match(TokenType.AND):
            right = self.parse_bitwise_or()
            left = BinaryExpr(op='&&', left=left, right=right, line=left.line)

        return left

    def parse_bitwise_or(self) -> Expression:
        """Parse bitwise OR expression."""
        left = self.parse_bitwise_xor()

        while self.match(TokenType.BIT_OR):
            right = self.parse_bitwise_xor()
            left = BinaryExpr(op='|', left=left, right=right, line=left.line)

        return left

    def parse_bitwise_xor(self) -> Expression:
        """Parse bitwise XOR expression."""
        left = self.parse_bitwise_and()

        while self.match(TokenType.BIT_XOR):
            right = self.parse_bitwise_and()
            left = BinaryExpr(op='^', left=left, right=right, line=left.line)

        return left

    def parse_bitwise_and(self) -> Expression:
        """Parse bitwise AND expression."""
        left = self.parse_equality()

        while self.match(TokenType.BIT_AND):
            right = self.parse_equality()
            left = BinaryExpr(op='&', left=left, right=right, line=left.line)

        return left

    def parse_equality(self) -> Expression:
        """Parse equality expression."""
        left = self.parse_comparison()

        while True:
            if self.match(TokenType.EQ):
                right = self.parse_comparison()
                left = BinaryExpr(op='==', left=left, right=right, line=left.line)
            elif self.match(TokenType.NE):
                right = self.parse_comparison()
                left = BinaryExpr(op='!=', left=left, right=right, line=left.line)
            else:
                break

        return left

    def parse_comparison(self) -> Expression:
        """Parse comparison expression."""
        left = self.parse_shift()

        while True:
            if self.match(TokenType.LT):
                right = self.parse_shift()
                left = BinaryExpr(op='<', left=left, right=right, line=left.line)
            elif self.match(TokenType.GT):
                right = self.parse_shift()
                left = BinaryExpr(op='>', left=left, right=right, line=left.line)
            elif self.match(TokenType.LE):
                right = self.parse_shift()
                left = BinaryExpr(op='<=', left=left, right=right, line=left.line)
            elif self.match(TokenType.GE):
                right = self.parse_shift()
                left = BinaryExpr(op='>=', left=left, right=right, line=left.line)
            else:
                break

        return left

    def parse_shift(self) -> Expression:
        """Parse shift expression."""
        left = self.parse_additive()

        while True:
            if self.match(TokenType.SHL):
                right = self.parse_additive()
                left = BinaryExpr(op='<<', left=left, right=right, line=left.line)
            elif self.match(TokenType.SHR):
                right = self.parse_additive()
                left = BinaryExpr(op='>>', left=left, right=right, line=left.line)
            else:
                break

        return left

    def parse_additive(self) -> Expression:
        """Parse additive expression."""
        left = self.parse_multiplicative()

        while True:
            if self.match(TokenType.PLUS):
                right = self.parse_multiplicative()
                left = BinaryExpr(op='+', left=left, right=right, line=left.line)
            elif self.match(TokenType.MINUS):
                right = self.parse_multiplicative()
                left = BinaryExpr(op='-', left=left, right=right, line=left.line)
            else:
                break

        return left

    def parse_multiplicative(self) -> Expression:
        """Parse multiplicative expression."""
        left = self.parse_unary()

        while True:
            if self.match(TokenType.STAR):
                right = self.parse_unary()
                left = BinaryExpr(op='*', left=left, right=right, line=left.line)
            elif self.match(TokenType.SLASH):
                right = self.parse_unary()
                left = BinaryExpr(op='/', left=left, right=right, line=left.line)
            elif self.match(TokenType.PERCENT):
                right = self.parse_unary()
                left = BinaryExpr(op='%', left=left, right=right, line=left.line)
            else:
                break

        return left

    def parse_unary(self) -> Expression:
        """Parse unary expression."""
        line = self.current().line

        if self.match(TokenType.NOT):
            operand = self.parse_unary()
            return UnaryExpr(op='!', operand=operand, line=line)

        if self.match(TokenType.MINUS):
            operand = self.parse_unary()
            return UnaryExpr(op='-', operand=operand, line=line)

        if self.match(TokenType.BIT_NOT):
            operand = self.parse_unary()
            return UnaryExpr(op='~', operand=operand, line=line)

        if self.match(TokenType.INC):
            operand = self.parse_unary()
            return UnaryExpr(op='++', operand=operand, line=line)

        if self.match(TokenType.DEC):
            operand = self.parse_unary()
            return UnaryExpr(op='--', operand=operand, line=line)

        return self.parse_postfix()

    def parse_postfix(self) -> Expression:
        """Parse postfix expression."""
        expr = self.parse_primary()

        while True:
            if self.match(TokenType.INC):
                expr = PostfixExpr(op='++', operand=expr, line=expr.line)
            elif self.match(TokenType.DEC):
                expr = PostfixExpr(op='--', operand=expr, line=expr.line)
            elif self.match(TokenType.LBRACKET):
                index = self.parse_expression()
                self.consume(TokenType.RBRACKET)
                if isinstance(expr, VarExpr):
                    expr = ArrayAccessExpr(array=expr.name, index=index, line=expr.line)
                else:
                    self.error("Array access on non-variable")
            elif self.match(TokenType.LPAREN):
                # Function call
                if not isinstance(expr, VarExpr):
                    self.error("Expected function name")

                args = []
                if not self.check(TokenType.RPAREN):
                    args.append(self.parse_expression())
                    while self.match(TokenType.COMMA):
                        args.append(self.parse_expression())

                self.consume(TokenType.RPAREN)
                expr = CallExpr(name=expr.name, args=args, line=expr.line)
            else:
                break

        return expr

    def parse_primary(self) -> Expression:
        """Parse primary expression."""
        token = self.current()

        if self.match(TokenType.NUMBER):
            return NumberExpr(value=token.value, line=token.line)

        if self.match(TokenType.CHAR):
            return NumberExpr(value=token.value, line=token.line)

        if self.match(TokenType.STRING):
            return StringExpr(value=token.value, line=token.line)

        if self.match(TokenType.IDENT):
            return VarExpr(name=token.value, line=token.line)

        if self.match(TokenType.LPAREN):
            expr = self.parse_expression()
            self.consume(TokenType.RPAREN)
            return expr

        self.error(f"Unexpected token: {token}")


# =============================================================================
# CODE GENERATOR
# =============================================================================

class CodeGenerator:
    """
    Generates MIPS assembly from the AST.

    Features:
    - Stack-based calling convention
    - Local variable allocation
    - Function parameters
    - Array support
    - All arithmetic, logical, and bitwise operators
    - Control flow with break/continue support
    """

    def __init__(self):
        self.output: List[str] = []
        self.data: List[str] = []
        self.label_count = 0
        self.string_count = 0
        self.vars: Dict[str, int] = {}  # var_name -> stack offset
        self.arrays: Dict[str, tuple] = {}  # array_name -> (offset, size)
        self.stack_offset = 0
        self.current_function = ""
        self.loop_stack: List[tuple] = []  # (continue_label, break_label)

    def new_label(self, prefix: str = "L") -> str:
        """Generate a unique label."""
        label = f"{prefix}{self.label_count}"
        self.label_count += 1
        return label

    def add_string(self, s: str) -> str:
        """Add a string to the data section and return its label."""
        label = f"str{self.string_count}"
        self.string_count += 1
        # Escape special characters for MIPS assembly
        escaped = s.replace('\\', '\\\\').replace('\n', '\\n').replace('\t', '\\t').replace('"', '\\"')
        self.data.append(f'{label}: .asciiz "{escaped}"')
        return label

    def emit(self, code: str) -> None:
        """Emit a line of assembly."""
        self.output.append(code)

    def generate(self, ast: Program) -> str:
        """Generate assembly for the entire program."""
        for func in ast.functions:
            self.gen_function(func)

        result = [".data"]
        result.extend(self.data)
        result.append("")
        result.append(".text")
        result.append(".globl main")
        result.append("")
        result.extend(self.output)

        # Built-in functions
        result.append("")
        result.append("# Built-in functions")
        result.append("print_int:")
        result.append("    li $v0, 1")
        result.append("    syscall")
        result.append("    jr $ra")
        result.append("")
        result.append("print_str:")
        result.append("    li $v0, 4")
        result.append("    syscall")
        result.append("    jr $ra")
        result.append("")
        result.append("print_char:")
        result.append("    li $v0, 11")
        result.append("    syscall")
        result.append("    jr $ra")
        result.append("")
        result.append("read_int:")
        result.append("    li $v0, 5")
        result.append("    syscall")
        result.append("    move $t0, $v0")
        result.append("    jr $ra")

        return "\n".join(result)

    def gen_function(self, func: Function) -> None:
        """Generate code for a function."""
        self.current_function = func.name
        self.vars = {}
        self.arrays = {}
        self.stack_offset = 0

        self.emit(f"{func.name}:")

        # Prologue: save return address and frame pointer
        self.emit("    addi $sp, $sp, -8")
        self.emit("    sw $ra, 4($sp)")
        self.emit("    sw $fp, 0($sp)")
        self.emit("    move $fp, $sp")

        # Allocate space for parameters
        # Parameters are passed in $a0-$a3 for the first 4, rest on stack
        for i, param in enumerate(func.params):
            self.emit("    addi $sp, $sp, -4")
            self.stack_offset += 4
            self.vars[param.name] = -self.stack_offset

            if i < 4:
                # Save from argument register
                self.emit(f"    sw $a{i}, {self.vars[param.name]}($fp)")
            else:
                # Load from caller's stack (above our frame)
                caller_offset = 8 + (i - 4) * 4
                self.emit(f"    lw $t0, {caller_offset}($fp)")
                self.emit(f"    sw $t0, {self.vars[param.name]}($fp)")

        # Generate function body
        for stmt in func.body:
            self.gen_statement(stmt)

        # Epilogue
        self.emit(f"{func.name}_exit:")
        self.emit("    move $sp, $fp")
        self.emit("    lw $fp, 0($sp)")
        self.emit("    lw $ra, 4($sp)")
        self.emit("    addi $sp, $sp, 8")
        self.emit("    jr $ra")
        self.emit("")

    def gen_statement(self, stmt: Statement) -> None:
        """Generate code for a statement."""
        if isinstance(stmt, Declaration):
            self.gen_declaration(stmt)
        elif isinstance(stmt, ExpressionStmt):
            self.gen_expression(stmt.expr)
        elif isinstance(stmt, IfStmt):
            self.gen_if(stmt)
        elif isinstance(stmt, WhileStmt):
            self.gen_while(stmt)
        elif isinstance(stmt, ForStmt):
            self.gen_for(stmt)
        elif isinstance(stmt, ReturnStmt):
            self.gen_return(stmt)
        elif isinstance(stmt, BreakStmt):
            self.gen_break(stmt)
        elif isinstance(stmt, ContinueStmt):
            self.gen_continue(stmt)

    def gen_declaration(self, decl: Declaration) -> None:
        """Generate code for a variable declaration."""
        if decl.array_size:
            # Array declaration
            size = decl.array_size * 4  # 4 bytes per int
            self.emit(f"    addi $sp, $sp, -{size}")
            self.stack_offset += size
            self.arrays[decl.name] = (-self.stack_offset, decl.array_size)
            self.vars[decl.name] = -self.stack_offset  # Points to first element
        else:
            # Regular variable
            self.emit("    addi $sp, $sp, -4")
            self.stack_offset += 4
            self.vars[decl.name] = -self.stack_offset

            if decl.init:
                self.gen_expression(decl.init)
                self.emit(f"    sw $t0, {self.vars[decl.name]}($fp)")

    def gen_if(self, stmt: IfStmt) -> None:
        """Generate code for an if statement."""
        else_label = self.new_label("else")
        end_label = self.new_label("endif")

        self.gen_expression(stmt.condition)
        self.emit(f"    beq $t0, $zero, {else_label}")

        for s in stmt.then_branch:
            self.gen_statement(s)

        self.emit(f"    j {end_label}")
        self.emit(f"{else_label}:")

        if stmt.else_branch:
            for s in stmt.else_branch:
                self.gen_statement(s)

        self.emit(f"{end_label}:")

    def gen_while(self, stmt: WhileStmt) -> None:
        """Generate code for a while loop."""
        loop_label = self.new_label("while")
        end_label = self.new_label("endwhile")

        self.loop_stack.append((loop_label, end_label))

        self.emit(f"{loop_label}:")
        self.gen_expression(stmt.condition)
        self.emit(f"    beq $t0, $zero, {end_label}")

        for s in stmt.body:
            self.gen_statement(s)

        self.emit(f"    j {loop_label}")
        self.emit(f"{end_label}:")

        self.loop_stack.pop()

    def gen_for(self, stmt: ForStmt) -> None:
        """Generate code for a for loop."""
        loop_label = self.new_label("for")
        continue_label = self.new_label("for_continue")
        end_label = self.new_label("endfor")

        self.loop_stack.append((continue_label, end_label))

        # Init
        if stmt.init:
            if isinstance(stmt.init, Declaration):
                self.gen_declaration(stmt.init)
            else:
                self.gen_expression(stmt.init)

        self.emit(f"{loop_label}:")

        # Condition
        if stmt.condition:
            self.gen_expression(stmt.condition)
            self.emit(f"    beq $t0, $zero, {end_label}")

        # Body
        for s in stmt.body:
            self.gen_statement(s)

        # Continue point for 'continue' statement
        self.emit(f"{continue_label}:")

        # Update
        if stmt.update:
            self.gen_expression(stmt.update)

        self.emit(f"    j {loop_label}")
        self.emit(f"{end_label}:")

        self.loop_stack.pop()

    def gen_return(self, stmt: ReturnStmt) -> None:
        """Generate code for a return statement."""
        if stmt.value:
            self.gen_expression(stmt.value)
            self.emit("    move $v0, $t0")
        self.emit(f"    j {self.current_function}_exit")

    def gen_break(self, stmt: BreakStmt) -> None:
        """Generate code for a break statement."""
        if not self.loop_stack:
            raise CodeGenError("'break' outside of loop", stmt.line)
        _, break_label = self.loop_stack[-1]
        self.emit(f"    j {break_label}")

    def gen_continue(self, stmt: ContinueStmt) -> None:
        """Generate code for a continue statement."""
        if not self.loop_stack:
            raise CodeGenError("'continue' outside of loop", stmt.line)
        continue_label, _ = self.loop_stack[-1]
        self.emit(f"    j {continue_label}")

    def gen_expression(self, expr: Expression) -> None:
        """Generate code for an expression. Result is in $t0."""
        if isinstance(expr, NumberExpr):
            self.emit(f"    li $t0, {expr.value}")

        elif isinstance(expr, StringExpr):
            label = self.add_string(expr.value)
            self.emit(f"    la $t0, {label}")

        elif isinstance(expr, VarExpr):
            if expr.name not in self.vars:
                raise CodeGenError(f"Undefined variable: {expr.name}", expr.line)
            offset = self.vars[expr.name]
            self.emit(f"    lw $t0, {offset}($fp)")

        elif isinstance(expr, ArrayAccessExpr):
            if expr.array not in self.arrays:
                raise CodeGenError(f"Undefined array: {expr.array}", expr.line)
            base_offset, _ = self.arrays[expr.array]

            # Calculate index
            self.gen_expression(expr.index)
            self.emit("    sll $t0, $t0, 2")  # Multiply by 4
            self.emit(f"    addi $t1, $fp, {base_offset}")
            self.emit("    add $t1, $t1, $t0")
            self.emit("    lw $t0, 0($t1)")

        elif isinstance(expr, AssignExpr):
            self.gen_expression(expr.value)

            if isinstance(expr.target, VarExpr):
                if expr.target.name not in self.vars:
                    raise CodeGenError(f"Undefined variable: {expr.target.name}", expr.line)
                offset = self.vars[expr.target.name]
                self.emit(f"    sw $t0, {offset}($fp)")
            elif isinstance(expr.target, ArrayAccessExpr):
                # Save the value
                self.emit("    addi $sp, $sp, -4")
                self.emit("    sw $t0, 0($sp)")

                # Calculate address
                if expr.target.array not in self.arrays:
                    raise CodeGenError(f"Undefined array: {expr.target.array}", expr.line)
                base_offset, _ = self.arrays[expr.target.array]
                self.gen_expression(expr.target.index)
                self.emit("    sll $t0, $t0, 2")
                self.emit(f"    addi $t1, $fp, {base_offset}")
                self.emit("    add $t1, $t1, $t0")

                # Store value
                self.emit("    lw $t0, 0($sp)")
                self.emit("    addi $sp, $sp, 4")
                self.emit("    sw $t0, 0($t1)")

        elif isinstance(expr, BinaryExpr):
            self.gen_binary(expr)

        elif isinstance(expr, UnaryExpr):
            self.gen_unary(expr)

        elif isinstance(expr, PostfixExpr):
            self.gen_postfix(expr)

        elif isinstance(expr, CallExpr):
            self.gen_call(expr)

    def gen_binary(self, expr: BinaryExpr) -> None:
        """Generate code for a binary expression."""
        # Short-circuit evaluation for && and ||
        if expr.op == '&&':
            end_label = self.new_label("and_end")
            self.gen_expression(expr.left)
            self.emit(f"    beq $t0, $zero, {end_label}")  # Short-circuit if false
            self.gen_expression(expr.right)
            self.emit("    sltu $t0, $zero, $t0")  # Convert to 0/1
            self.emit(f"{end_label}:")
            return

        if expr.op == '||':
            end_label = self.new_label("or_end")
            self.gen_expression(expr.left)
            self.emit(f"    bne $t0, $zero, {end_label}")  # Short-circuit if true
            self.gen_expression(expr.right)
            self.emit(f"{end_label}:")
            self.emit("    sltu $t0, $zero, $t0")  # Convert to 0/1
            return

        # Evaluate left operand
        self.gen_expression(expr.left)
        self.emit("    addi $sp, $sp, -4")
        self.emit("    sw $t0, 0($sp)")

        # Evaluate right operand
        self.gen_expression(expr.right)
        self.emit("    move $t1, $t0")

        # Load left operand
        self.emit("    lw $t0, 0($sp)")
        self.emit("    addi $sp, $sp, 4")

        # Generate operation
        op = expr.op
        if op == '+':
            self.emit("    add $t0, $t0, $t1")
        elif op == '-':
            self.emit("    sub $t0, $t0, $t1")
        elif op == '*':
            self.emit("    mul $t0, $t0, $t1")
        elif op == '/':
            self.emit("    div $t0, $t1")
            self.emit("    mflo $t0")
        elif op == '%':
            self.emit("    div $t0, $t1")
            self.emit("    mfhi $t0")
        elif op == '<':
            self.emit("    slt $t0, $t0, $t1")
        elif op == '>':
            self.emit("    slt $t0, $t1, $t0")
        elif op == '<=':
            self.emit("    slt $t0, $t1, $t0")
            self.emit("    xori $t0, $t0, 1")
        elif op == '>=':
            self.emit("    slt $t0, $t0, $t1")
            self.emit("    xori $t0, $t0, 1")
        elif op == '==':
            self.emit("    sub $t0, $t0, $t1")
            self.emit("    sltiu $t0, $t0, 1")  # 1 if t0 == 0
        elif op == '!=':
            self.emit("    sub $t0, $t0, $t1")
            self.emit("    sltu $t0, $zero, $t0")  # 1 if t0 != 0
        elif op == '&':
            self.emit("    and $t0, $t0, $t1")
        elif op == '|':
            self.emit("    or $t0, $t0, $t1")
        elif op == '^':
            self.emit("    xor $t0, $t0, $t1")
        elif op == '<<':
            self.emit("    sllv $t0, $t0, $t1")
        elif op == '>>':
            self.emit("    srav $t0, $t0, $t1")

    def gen_unary(self, expr: UnaryExpr) -> None:
        """Generate code for a unary expression."""
        if expr.op == '++':
            # Pre-increment
            if isinstance(expr.operand, VarExpr):
                offset = self.vars[expr.operand.name]
                self.emit(f"    lw $t0, {offset}($fp)")
                self.emit("    addi $t0, $t0, 1")
                self.emit(f"    sw $t0, {offset}($fp)")
            else:
                raise CodeGenError("Cannot increment non-variable", expr.line)

        elif expr.op == '--':
            # Pre-decrement
            if isinstance(expr.operand, VarExpr):
                offset = self.vars[expr.operand.name]
                self.emit(f"    lw $t0, {offset}($fp)")
                self.emit("    addi $t0, $t0, -1")
                self.emit(f"    sw $t0, {offset}($fp)")
            else:
                raise CodeGenError("Cannot decrement non-variable", expr.line)

        elif expr.op == '-':
            # Negation
            self.gen_expression(expr.operand)
            self.emit("    sub $t0, $zero, $t0")

        elif expr.op == '!':
            # Logical NOT
            self.gen_expression(expr.operand)
            self.emit("    sltiu $t0, $t0, 1")  # 1 if t0 == 0, else 0

        elif expr.op == '~':
            # Bitwise NOT
            self.gen_expression(expr.operand)
            self.emit("    nor $t0, $t0, $zero")

    def gen_postfix(self, expr: PostfixExpr) -> None:
        """Generate code for a postfix expression."""
        if not isinstance(expr.operand, VarExpr):
            raise CodeGenError("Cannot apply postfix operator to non-variable", expr.line)

        offset = self.vars[expr.operand.name]
        self.emit(f"    lw $t0, {offset}($fp)")
        self.emit("    move $t2, $t0")  # Save original value

        if expr.op == '++':
            self.emit("    addi $t0, $t0, 1")
        else:
            self.emit("    addi $t0, $t0, -1")

        self.emit(f"    sw $t0, {offset}($fp)")
        self.emit("    move $t0, $t2")  # Return original value

    def gen_call(self, expr: CallExpr) -> None:
        """Generate code for a function call."""
        # Built-in functions
        if expr.name in ('print_int', 'print_str', 'print_char', 'read_int'):
            if expr.args:
                self.gen_expression(expr.args[0])
                self.emit("    move $a0, $t0")
            self.emit(f"    jal {expr.name}")
            return

        # Save caller-saved registers
        self.emit("    addi $sp, $sp, -4")
        self.emit("    sw $t0, 0($sp)")

        # Push arguments (in reverse order for stack-based passing)
        # First 4 args go in $a0-$a3, rest on stack
        for i, arg in enumerate(expr.args):
            self.gen_expression(arg)
            if i < 4:
                self.emit(f"    move $a{i}, $t0")
            else:
                self.emit("    addi $sp, $sp, -4")
                self.emit("    sw $t0, 0($sp)")

        # Call function
        self.emit(f"    jal {expr.name}")

        # Clean up stack arguments
        if len(expr.args) > 4:
            cleanup = (len(expr.args) - 4) * 4
            self.emit(f"    addi $sp, $sp, {cleanup}")

        # Move result to $t0
        self.emit("    move $t0, $v0")

        # Restore caller-saved register
        self.emit("    lw $t1, 0($sp)")
        self.emit("    addi $sp, $sp, 4")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def compile_file(input_path: str, output_path: str = None) -> str:
    """Compile a C file to MIPS assembly."""
    with open(input_path, 'r') as f:
        source = f.read()

    # Lexical analysis
    lexer = Lexer(source, input_path)
    tokens = lexer.tokenize()

    # Parsing
    parser = Parser(tokens)
    ast = parser.parse()

    # Code generation
    codegen = CodeGenerator()
    assembly = codegen.generate(ast)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(assembly)

    return assembly


def main():
    """Main entry point for the compiler."""
    if len(sys.argv) < 2:
        print("LiteCC - A Lightweight C to MIPS Compiler")
        print("")
        print("Usage: litecc <input.c> [-o <output.asm>]")
        print("")
        print("Options:")
        print("  -o <file>    Write output to <file> (default: out.asm)")
        print("  --tokens     Print tokens and exit")
        print("  --ast        Print AST and exit")
        print("  --help       Show this help message")
        sys.exit(1)

    input_file = None
    output_file = "out.asm"
    print_tokens = False
    print_ast = False

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-o':
            i += 1
            if i >= len(sys.argv):
                print("Error: -o requires an argument")
                sys.exit(1)
            output_file = sys.argv[i]
        elif arg == '--tokens':
            print_tokens = True
        elif arg == '--ast':
            print_ast = True
        elif arg == '--help':
            main()  # Print help
        elif not arg.startswith('-'):
            input_file = arg
        else:
            print(f"Error: Unknown option '{arg}'")
            sys.exit(1)
        i += 1

    if not input_file:
        print("Error: No input file specified")
        sys.exit(1)

    try:
        with open(input_file, 'r') as f:
            source = f.read()

        lexer = Lexer(source, input_file)
        tokens = lexer.tokenize()

        if print_tokens:
            for token in tokens:
                print(token)
            sys.exit(0)

        parser = Parser(tokens)
        ast = parser.parse()

        if print_ast:
            import json
            from dataclasses import asdict
            def ast_to_dict(node):
                if isinstance(node, list):
                    return [ast_to_dict(n) for n in node]
                elif hasattr(node, '__dataclass_fields__'):
                    return {k: ast_to_dict(v) for k, v in asdict(node).items()}
                return node
            print(json.dumps(ast_to_dict(ast), indent=2))
            sys.exit(0)

        codegen = CodeGenerator()
        assembly = codegen.generate(ast)

        with open(output_file, 'w') as f:
            f.write(assembly)

        print(f"Compiled {input_file} -> {output_file}")

    except CompilerError as e:
        print(e)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Internal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
