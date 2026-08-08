from collections import deque


class PlayerTracker:
    """
    Track player movement using detected center positions.

    This version does not identify the player between frames.
    It only records recent positions.
    """


    def __init__(self, history_size=30):
        """
        Args:
            history_size:
                Number of previous positions to keep.
        """

        self.positions = deque(
            maxlen=history_size
        )



    def update(self, x, y):
        """
        Add a new player position.

        Args:
            x:
                Player center x coordinate

            y:
                Player center y coordinate
        """

        self.positions.append(
            (x, y)
        )



    def get_current_position(self):
        """
        Return the latest player position.
        """

        if len(self.positions) == 0:
            return None

        return self.positions[-1]