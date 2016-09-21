import unittest
import textcnn as tc

class MajorityTests(unittest.TestCase):
    def majority_test_helper(self, stream, n, expectedCounts):
        counts = tc.majority.majority_elements(stream, n)
        self.assertEqual(expectedCounts, counts)

    def test_majority(self):
        stream = ['a', 'b', 'a', 'c', 'b', 'c', 'b', 'c', 'c']
        n = 2
        expectedCounts = {'b': 2, 'c': 3}
        self.majority_test_helper(stream, n, expectedCounts)

    def test_majority2(self):
        stream = ['a', 'b', 'c', 'd', 'a', 'a', 'a', 'a', 'a']
        n = 2
        expectedCounts = {'a': 5}
        self.majority_test_helper(stream, n, expectedCounts)

if __name__ == "__main__":
    unittest.main()
