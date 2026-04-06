import numpy as np

class KnapsackProblem:
    """
    A class to solve the 0-1 Knapsack problem using dynamic programming.

    Attributes:
        weights (list): A list of item weights.
        values (list): A list of item values.
        capacity (int): The maximum weight capacity of the knapsack.
        n (int): The number of items.
    """

    def __init__(self, weights, values, capacity):
        """
        Initializes the Knapsack problem with weights, values, and capacity.

        Args:
            weights (list): A list of item weights.
            values (list): A list of item values.
            capacity (int): The maximum weight capacity of the knapsack.
        """
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.n = len(weights)
        # Create a DP table
        self.dp = np.zeros((self.n+1, self.capacity+1)) # [[0 for _ in range(self.capacity + 1)] for _ in range(self.n + 1)]

    def solve(self):
        """
        Solves the 0-1 Knapsack problem using dynamic programming.

        Returns:
            int: The maximum total value that can be accommodated in the knapsack.
        """
        # Fill the DP table
        for i in range(1, self.n + 1):
            for w in range(self.capacity + 1):
                if self.weights[i - 1] <= w:
                    self.dp[i][w] = max(self.dp[i - 1][w], self.values[i - 1] + self.dp[i - 1][w - self.weights[i - 1]])
                else:
                    self.dp[i][w] = self.dp[i - 1][w]

        print(self.dp)

        return self.dp[self.n][self.capacity]

    def find_selected_items(self):
        """
        Reconstructs and returns the list of item indices selected in the optimal solution
        to the 0-1 Knapsack problem using the dynamic programming table.

        This function traces back through the dynamic programming (DP) table to identify
        which items were included in the knapsack to achieve the maximum total value.

        Returns:
            list[int]: A list of indices (0-based) of the selected items that make up the
                       optimal solution. The indices correspond to positions in the original
                       weights and values lists.
        """
        selected_items = []
        w = self.capacity
        for i in range(self.n, 0, -1):
            # If the value differs with and without the i-th item, it was selected
            if self.dp[i][w] != self.dp[i - 1][w]:
                selected_items.append(i - 1)  # item index (0-based)
                w -= self.weights[i - 1]  # reduce remaining capacity
        return selected_items