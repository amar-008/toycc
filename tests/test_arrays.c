// Test array operations
int main() {
    int arr[5];
    int i = 0;

    // Initialize array
    for (i = 0; i < 5; i++) {
        arr[i] = i + 1;
    }

    // Print array elements
    for (i = 0; i < 5; i++) {
        print_int(arr[i]);
        print_str("\n");
    }

    // Sum array
    int sum = 0;
    for (i = 0; i < 5; i++) {
        sum = sum + arr[i];
    }
    print_int(sum);  // 15
    print_str("\n");

    return 0;
}
