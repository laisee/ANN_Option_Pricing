class VectorClock:
    def __init__(self, node_id, num_nodes):
        """
        Initialize a vector clock with a given node ID and number of nodes.

        :param node_id: The ID of the node this clock belongs to.
        :param num_nodes: The total number of nodes in the system.
        """
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.clock = [0] * num_nodes

    def increment(self):
        """
        Increment the clock value for the current node.
        """
        self.clock[self.node_id] += 1

    def update(self, other_clock):
        """
        Update this clock with the values from another clock.

        :param other_clock: The other clock to update from.
        """
        for i in range(self.num_nodes):
            self.clock[i] = max(self.clock[i], other_clock.clock[i])

    def __str__(self):
        """
        Return a string representation of the clock.
        """
        return str(self.clock)

    def __lt__(self, other_clock):
        """
        Check if this clock is less than another clock.

        :param other_clock: The other clock to compare with.
        :return: True if this clock is less than the other clock, False otherwise.
        """
        return all(x < y for x, y in zip(self.clock, other_clock.clock))

    def __le__(self, other_clock):
        """
        Check if this clock is less than or equal to another clock.

        :param other_clock: The other clock to compare with.
        :return: True if this clock is less than or equal to the other clock, False otherwise.
        """
        return all(x <= y for x, y in zip(self.clock, other_clock.clock))

    def __eq__(self, other_clock):
        """
        Check if this clock is equal to another clock.

        :param other_clock: The other clock to compare with.
        :return: True if this clock is equal to the other clock, False otherwise.
        """
        return self.clock == other_clock.clock


# Create two vector clocks with 3 nodes each
clock1 = VectorClock(0, 3)
clock2 = VectorClock(1, 3)

# Increment clock1 a few times
clock1.increment()
clock1.increment()
clock2.increment()
clock2.increment()
clock2.increment()

# Update clock2 with clock1's values
clock2.update(clock1)

# Print the clock values
print("Clock 1:", clock1)
print("Clock 2:", clock2)

# Compare the clocks
print("Clock 1 < Clock 2:", clock1 < clock2)
print("Clock 1 <= Clock 2:", clock1 <= clock2)
print("Clock 1 == Clock 2:", clock1 == clock2)
