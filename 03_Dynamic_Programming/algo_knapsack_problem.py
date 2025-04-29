
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

    def solve(self):
        """
        Solves the 0-1 Knapsack problem using dynamic programming.

        Returns:
            int: The maximum total value that can be accommodated in the knapsack.
        """
        pass

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
        pass