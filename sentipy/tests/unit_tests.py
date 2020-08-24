import unittest
from sentipy.utils import process_patterns, eliminate_stopwords


class UtilsTest(unittest.TestCase):
    """
    Class to test functions in utils
    """

    def test_process_patterns(self):
        """
        Tests process_patterns function
        """

        text_raw = 'This is a $100 question. You have 2 attempts as on 21/10/2019'
        text_del = 'This is a  question. You have  attempts as on '
        text_repl = 'This is a _currency_ question. You have _number_ attempts as on _date_'

        # Testing with default option of delete=True
        self.assertEqual(process_patterns(text_raw,False), text_del)

        # Testing with option of delete=False
        self.assertEqual(process_patterns(text_raw, False, False), text_repl)

    def test_eliminate_stopwords(self):
        """
        Tests eliminate_stopwrds function
        """

        text_raw = 'This is a $100 question. You have 2 attempts as on 21/10/2019'
        text_wo_stops = '$ 100 question . 2 attempts 21/10/2019'

        self.assertEqual(eliminate_stopwords(text_raw), text_wo_stops)


if __name__ == "__main__":
    unittest.main()
