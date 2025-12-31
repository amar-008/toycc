// Test bitwise operators
int main() {
    int a = 5;   // 0101
    int b = 3;   // 0011

    // AND
    print_int(a & 7);  // 5 (0101 & 0111)
    print_str("\n");

    // OR
    print_int(a | b);  // 7 (0101 | 0011)
    print_str("\n");

    // XOR
    print_int(a ^ b);  // 6 (0101 ^ 0011)
    print_str("\n");

    // NOT
    print_int(~a);  // -6 (in two's complement)
    print_str("\n");

    // Left shift
    print_int(a << 2);  // 20 (5 * 4)
    print_str("\n");

    // Right shift
    print_int(8 >> 2);  // 2 (8 / 4)
    print_str("\n");

    return 0;
}