// Test function definitions and calls

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int add(int a, int b) {
    return a + b;
}

int fibonacci(int n) {
    if (n <= 1) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

int main() {
    // Test factorial
    print_int(factorial(5));  // 120
    print_str("\n");

    // Test add
    print_int(add(3, 5));  // 8
    print_str("\n");

    // Test fibonacci
    print_int(fibonacci(10));  // 55
    print_str("\n");

    return 0;
}
