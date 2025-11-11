import java.util.*;

// Node structure for Huffman tree
class Node {
    char ch;
    int freq;
    Node left, right;

    Node(char ch, int freq) {
        this.ch = ch;
        this.freq = freq;
    }

    Node(int freq, Node left, Node right) {
        this.ch = '\0';
        this.freq = freq;
        this.left = left;
        this.right = right;
    }
}

public class DAA_Huffman {

    // Recursive function to generate Huffman codes
    public static void generateCodes(Node root, String code, Map<Character, String> huffmanCode) {
        if (root == null)
            return;

        // Leaf node — store the code
        if (root.left == null && root.right == null) {
            huffmanCode.put(root.ch, code);
            return;
        }

        generateCodes(root.left, code + "0", huffmanCode);
        generateCodes(root.right, code + "1", huffmanCode);
    }

    public static void main(String[] args) {
        String text = "huffman coding example";

        // Count frequency of each character
        Map<Character, Integer> freq = new HashMap<>();
        for (char ch : text.toCharArray())
            freq.put(ch, freq.getOrDefault(ch, 0) + 1);

        // Priority queue (min-heap) for building Huffman tree
        PriorityQueue<Node> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a.freq));

        // Create a leaf node for each character
        for (Map.Entry<Character, Integer> entry : freq.entrySet())
            pq.add(new Node(entry.getKey(), entry.getValue()));

        // Build the Huffman tree
        while (pq.size() > 1) {
            Node left = pq.poll();
            Node right = pq.poll();

            Node parent = new Node(left.freq + right.freq, left, right);
            pq.add(parent);
        }

        // Root of Huffman Tree
        Node root = pq.peek();

        // Generate Huffman codes
        Map<Character, String> huffmanCode = new HashMap<>();
        generateCodes(root, "", huffmanCode);

        // Print the codes
        System.out.println("Huffman Codes:");
        for (Map.Entry<Character, String> entry : huffmanCode.entrySet())
            System.out.println(entry.getKey() + " : " + entry.getValue());

        // Encode the input text
        StringBuilder encoded = new StringBuilder();
        for (char ch : text.toCharArray())
            encoded.append(huffmanCode.get(ch));

        System.out.println("\nOriginal text: " + text);
        System.out.println("Encoded text: " + encoded);
    }
}