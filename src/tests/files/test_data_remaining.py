import unittest

from src import app
from pages import map
from processing import data

class test_data_remaining_code(unittest.TestCase):

    def test_data(self):
        self.assertListEqual(data.get_morphology_names(), [])
        self.assertFalse(data.isDownloading())