import unittest

from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

from src import app
from src.pages import analysis

class test_analysis_page(unittest.TestCase):

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

    def test_show_progress_bar(self):
        c = self.context(self)
        c.func = analysis.show_progress_bar
        c.p = dict(
            n_clicks = None,
            n_intervals = None,
            value = 101,
            ceil = None,
            steps = None
        )

        c.eq(True)
        
        c.p["steps"] = "1"
        c.p["ceil"] = "1"
        c.eq(True)

        c.p["n_clicks"] = 1
        c.eq(True)

        c.p["value"] = 0
        c.eq(True)


    def test_progress_bar(self):
        c = self.context(self)
        c.func = analysis.progress_bar
        c.p = dict(
            n_intervals = None,
            value = 100,
            ceil = None,
            prev_value = "-1",
            prev_label = "1"
        )

        c.te(None, None)

        c.p["value"] = 0
        c.te("-1", "1")

        c.p["ceil"] = 1
        c.te(2, "2%")


    def test_update_threshold_sim(self):
        c = self.context(self)
        c.func = analysis.update_threshold_simulation
        c.p = dict(
            process = None,
            lx = None,
            ly = None,
            ceil = None,
            steps = None,
            prev_fig = None
        )

        c.trigger = "process"
        c.eq(None)

        c.p["ceil"] = 1
        c.p["steps"] = 1
        c.tr(not None)

        c.trigger = ""
        fig = c.ctx.run(c.run_context)
        self.assertEqual(fig["layout"]["xaxis"]["type"], "linear")
        self.assertEqual(fig["layout"]["yaxis"]["type"], "linear")

        c.p["lx"] = 1
        c.p["ly"] = 1
        fig = c.ctx.run(c.run_context)
        self.assertEqual(fig["layout"]["xaxis"]["type"], "log")
        self.assertEqual(fig["layout"]["yaxis"]["type"], "log")


    def test_update_distance_graph(self):
        c = self.context(self)
        c.func = analysis.update_distance_graph
        c.p = dict(
            log_x = None,
            log_y = None,
            catalog = "FR0",
            current_figure = None
        )

        fig = c.ctx.run(c.run_context)
        self.assertEqual(fig["layout"]["xaxis"]["type"], "log")
        self.assertEqual(fig["layout"]["yaxis"]["type"], "log")

        c.trigger = "log_x"
        c.p["log_x"] = 1
        c.p["log_y"] = 1
        fig = c.ctx.run(c.run_context)
        self.assertEqual(fig["layout"]["xaxis"]["type"], "linear")
        self.assertEqual(fig["layout"]["yaxis"]["type"], "linear")

        
    def test_misc(self):
        c = self.context(self)
        
        c.func = analysis.update_sunburst
        c.p = dict(dummy=None)
        c.tr(not None)

        c.func = analysis.update_snapshot_002
        c.p = dict(
            threshold = 1,
            venn1 = "MiraBest",
            venn2 = "MiraBest",
        )
        fig, title = c.ctx.run(c.run_context)
        self.assertEqual(title, 'Percentage Duplicates:\tMiraBest(27.1%)\tMiraBest(27.1%)')
        self.assertIsNotNone(fig)

        c.func = analysis.update_type_graph
        c.p = dict(dummy=None)
        c.tr(not None)

        c.func = analysis.update_options
        c.p = dict(n_clicks=None)
        c.tr(not None)