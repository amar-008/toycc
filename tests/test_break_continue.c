// Test break and continue statements
int main() {
    int i = 0;

    // Test continue: skip 3
    for (i = 1; i <= 5; i++) {
        if (i == 3) {
            continue;
        }
        print_int(i);
        print_str("\n");
    }

    // Test break: sum until sum >= 25
    int sum = 0;
    i = 1;
    while (i <= 100) {
        sum = sum + i;
        if (sum >= 25) {
            break;
        }
        i++;
    }
    // 1+2+3+4+5+6+7 = 28, but loop: sum=1,2,6,10,15,21,28; at 28>=25, break
    // Actually the loop is: sum=0+1=1 (<25), i=2; sum=1+2=3, i=3; sum=3+3=6, i=4; sum=6+4=10, i=5;
    // sum=10+5=15, i=6; sum=15+6=21, i=7; sum=21+7=28>=25, break
    // Output: 28
    // Wait, the increment happens after the break check, so:
    // i=1: sum=1, not >=25, i++->2
    // i=2: sum=3, not >=25, i++->3
    // i=3: sum=6, not >=25, i++->4
    // i=4: sum=10, not >=25, i++->5
    // i=5: sum=15, not >=25, i++->6
    // i=6: sum=21, not >=25, i++->7
    // i=7: sum=28, >=25, break
    // Final sum = 28
    // Let me make it simpler: output 1+2+3+4+5+6+7 = 28
    // Or change condition to sum > 20 to get 21
    // Actually let's just compute: when sum >= 25, we have 1+2+3+4+5+6+7 = 28, so output 28
    // But actually let's output 25 exactly by using a different approach
    // Simplest: change to sum to 5 iterations: sum = 1+2+3+4+5 = 15, break when i>5
    // Let me just output what the current code would output
    print_int(sum);
    print_str("\n");

    return 0;
}
