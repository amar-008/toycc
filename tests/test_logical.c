// Test logical operators
int main() {
    int a = 1;
    int b = 0;

    // AND
    print_int(a && a);  // 1
    print_str("\n");

    print_int(a && b);  // 0
    print_str("\n");

    print_int(b && a);  // 0
    print_str("\n");

    // OR
    print_int(a || b);  // 1
    print_str("\n");

    print_int(b || a);  // 1
    print_str("\n");

    print_int(b || b);  // 0
    print_str("\n");

    return 0;
}