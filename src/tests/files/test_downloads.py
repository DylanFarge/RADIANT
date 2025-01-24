import unittest

from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict
from pages import map

from src import app
from src.pages import downloads

class test_downloads_page(unittest.TestCase):

    class context():

        def __init__(self,testcase):
            self.p = {}
            self.func = None
            self.trigger = ""
            self.ctx = copy_context()
            self.ts = testcase

        def run_context(self):
            context_value.set(AttributeDict(**{"triggered_inputs":[{"prop_id":self.trigger}]}))
            return self.func(**self.p)
        
        def eq(self, result):
            self.ts.assertEqual(self.ctx.run(self.run_context), result)

        def te(self, *result):
            self.ts.assertTupleEqual(self.ctx.run(self.run_context), result)

        def tr(self, result):
            self.ts.assertTrue(self.ctx.run(self.run_context), result)
    
    def test_update_priority_dropdown(self):
        c = self.context(self)
        c.func = downloads.update_priority_dropdown
        c.p = dict(
            rm = None,
            catalogs = None,
            curr = None,
        )

        c.te(True, [], [])

        c.p["rm"] = True
        c.p["curr"] = ["FIRST", "SECOND", "THIRD"]
        c.p["catalogs"] = ["FIRST"]
        c.te(False, ["FIRST"], ["FIRST"])


    def test_show_progress_bar(self):
        c = self.context(self)
        c.func = downloads.show_progress_bar
        c.p = dict(
            n_clicks = None,
            n_intervals = None,
            progress = None,
        )

        c.eq(True)

    
    def test_progress_bar(self):
        c = self.context(self)
        c.func = downloads.progress_bar
        c.p = dict(
            n_intervals = None,
            num_of_sources = None,
            image_limit = None,
            surveys = None,
            progress = 101,
        )

        c.te(None, None, False, None)

        c.p["progress"] = None
        c.te(None, None, False, None)

        c.p["num_of_sources"]  = "Source: 100"
        c.p["surveys"] = ["NVSS"]
        c.te(2, "2%", False, None)

        c.p["image_limit"] = True
        c.te(2, "2%", False, None)


    def test_download(self):
        c = self.context(self)
        c.func = downloads.download
        c.p = dict(
            n_clicks = None,
            catalogs = None,
            types = None,
            include_images = None,
            surveys = None,
            image_limit = None,
            remove_duplicates = None,
            threshold = None,
            priority = [],
        )

        c.eq(None)

        c.p["n_clicks"] = 1
        c.eq(None)

        c.p["catalogs"] = ["FR0"]
        c.p["types"] = ["COMPACT"]
        c.p["remove_duplicates"] = True
        c.eq(None)

        c.p["threshold"] = 1
        c.tr(not None)

        c.p["include_images"] = True
        c.p["surveys"] = ["NVSS","CO"]
        c.p["image_limit"] = 2
        c.tr(not None)

        c.p["include_images"] = False
        c.p["catalogs"] = ["FR0", "FRI", "FRII"]
        c.p["priority"] = ["FR0","FRI"]
        c.tr(not None)

    def test_display_num_of_sources_to_download(self):
        c = self.context(self)
        c.func = downloads.display_num_of_sources_to_download
        c.p = dict(
            catalogs = None,
            types = None,
            threshold = 0,
            prior = None,
        )

        c.eq("Sources: 0")

        c.p["catalogs"] = ["MiraBest"]
        c.p["types"] = ["COMPACT"]
        c.eq("Sources: 108")


    def test_select_all_morphologies(self):
        c = self.context(self)
        c.func = downloads.select_all_morphologies
        c.p = dict(
            n_clicks = None,
            catalogs = ["FRI"],
            curr = None,
        )

        c.te([], ["FRI"])

        c.p["curr"] = ["FRI", "FRII"]
        c.te(["FRI"], ["FRI"])

        c.trigger = "select-all-morphologies.nclicks"
        c.te(["FRI"],["FRI"])


    def test_misc(self):
        c = self.context(self)
        
        c.func = downloads.get_catalog_options
        c.p = dict(dummy = None)
        c.tr(not None)

        c.func = downloads.select_all_catalogs
        c.p = dict(n_clicks = None)
        c.eq(None)
        c.p["n_clicks"] = 1
        c.tr(not None)

        c.func = downloads.toggle_duplicate_options
        c.p = dict(value = 0)
        c.te(True, None)
        c.p["value"] = 1
        c.te(False, None)

        c.func = downloads.toggle_image_options
        c.p = dict(value = 0)
        c.te(True, None, True, None)
        c.p["value"] = 1
        c.te(False, None, False, None)