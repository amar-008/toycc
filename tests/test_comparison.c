// Test comparison operators
int main() {
    int a = 10;
    int b = 20;

    // Less than
    print_int(a < b);  // 1
    print_str("\n");

    print_int(b < a);  // 0
    print_str("\n");

    // Greater than
    print_int(b > a);  // 1
    print_str("\n");

    // Less than or equal
    print_int(a <= 10);  // 1
    print_str("\n");

    // Greater than or equal
    print_int(a >= 20);  // 0
    print_str("\n");

    // Equal
    print_int(a == 10);  // 1
    print_str("\n");

    return 0;
}