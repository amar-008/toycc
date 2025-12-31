// Test loop constructs
int main() {
    int sum = 0;
    int i = 0;

    // While loop: sum 1 to 10
    i = 1;
    while (i <= 10) {
        sum = sum + i;
        i++;
    }
    print_int(sum);  // 55
    print_str("\n");

    // For loop: factorial of 5
    int fact = 1;
    for (i = 1; i <= 5; i++) {
        fact = fact * i;
    }
    print_int(fact);  // 120
    print_str("\n");

    // Nested loops test
    int count = 0;
    for (i = 0; i < 5; i++) {
        int j = 0;
        while (j < 2) {
            count++;
            j++;
        }
    }
    print_int(count);  // 10
    print_str("\n");

    return 0;
}