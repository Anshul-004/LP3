import java.util.Scanner;

public class DAA_Fibonacci {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter number of terms: ");
        int n = sc.nextInt();

        int first = 0, second = 1;
        int stepCount = 0; // To count steps

        System.out.println("Fibonacci Series:");

        for (int i = 1; i <= n; i++) {
            System.out.print(first + " ");
            stepCount++; 

            int next = first + second;
            stepCount++; 
            first = second;
            second = next;
            stepCount += 2;
        }

        System.out.println("\nTotal steps executed: " + stepCount);
        sc.close();
    }
}
