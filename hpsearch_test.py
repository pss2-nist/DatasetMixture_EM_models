import unittest
from HyperparameterSearch import *


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.parser = create_parser()

    def test_example(self):
        path1 = ""
        path2 = ""
        parsed = self.parser.parse_args(['--dataset1', path1, '--dataset2', path2])
        # self.assertEqual(parsed.dataset1, "")
        print(self.dataset1)

if __name__=="__main__":
    unittest.main()