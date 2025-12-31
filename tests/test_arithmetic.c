// Test arithmetic operations
int main() {
    int a = 10;
    int b = 5;
    int c = 20;

    // Addition
    print_int(a + b + c + c);  // 55
    print_str("\n");

    // Subtraction
    print_int(a - b - c + 5);  // -10
    print_str("\n");

    // Multiplication
    print_int(a * b * 12);  // 600
    print_str("\n");

    // Division
    print_int(c / b - 1);  // 3
    print_str("\n");

    // Modulo
    print_int(17 % 5);  // 2
    print_str("\n");

    return 0;
}
