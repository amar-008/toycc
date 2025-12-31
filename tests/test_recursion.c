// Test recursion
int fib(int n) {
    if (n <= 1) {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

int fact(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * fact(n - 1);
}

int main() {
    print_int(fib(10));  // 55
    print_str("\n");

    print_int(fact(5));  // 120
    print_str("\n");

    return 0;
}