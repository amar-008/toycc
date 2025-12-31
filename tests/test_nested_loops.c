// Test nested loops
int main() {
    int i = 0;
    int j = 0;

    // Multiplication table 3x3
    for (i = 1; i <= 3; i++) {
        for (j = 1; j <= 3; j++) {
            print_int(i * j);
            print_str("\n");
        }
    }

    return 0;
}
