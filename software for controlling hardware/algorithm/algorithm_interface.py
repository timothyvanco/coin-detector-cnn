
class AlgorithmInterface:
    """Common interface for all algorithms"""

    def alg(self, iterations: int) -> dict:
        """
        Method handling a specific algorithm implementation.
        Params:
            iterations - Number of times the algorithm has already run. For output purposes.
        Return:
            (dict)
            {
                found: bool - Indicates whether anything has been found or not
                image_annotated: np - Numpy image in RGB (not BGR!), which will be shown on the display. May be annotated.
                image_original: np - Numpy image in RGB (not BGR!), which will be shown on the display. May NOT be annotated.
                coins: (dict)
                {
                    ' 1Kc': 0, 
                    ' 2Kc': 0, 
                    ' 5Kc': 0,
                    '10Kc': 0,
                    '20Kc': 0,
                    '50Kc': 0,
                    '  1c': 0,
                    '  2c': 0,
                    '  5c': 0,
                    ' 10c': 0,
                    ' 20c': 0,
                    ' 50c': 0,
                    '1EUR': 0,
                    '2EUR': 0
                } - Found coin values
            }
        """
        raise Exception('Do not use an interface directly. Implement it.')