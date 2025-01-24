import unittest
import os

from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

from dash import html

from src import app
from pages import map

class homepage_test(unittest.TestCase):

    def test_remove_catalog(self):

        trigger = ""
        params = dict(
            delete_btn = None,
            rm_cats = None,
            upload = None,
            delete_options = None,
            ra = None,
            dec = None,
            types = None,
            contents = None,
            filename = None
        )
        ctx = copy_context()

        def run_callback():
            context_value.set(AttributeDict(**{"triggered_inputs":[{"prop_id":trigger}]}))
            return map.remove_catalog(**params)
        
        def te(*result):
            self.assertTupleEqual(ctx.run(run_callback), result)
        # -------------------------------------------------------------------------------

        trigger = "delete_catalogs.value"
        te(None, True)
        params["rm_cats"] = ["1"]
        te(None, False)

        trigger = "upload-button.nclicks"
        te(None, True)
        params['ra'] = "1"
        params['dec'] = "1"
        params['types'] = "1"
        params['contents'] = "1"
        params['filename'] = "mock.name"
        params["delete_options"] = []
        te(['mock'], False)

        if os.path.exists("catalogs/uploaded"):
            if len(os.listdir("catalogs/uploaded")) != 0:
                raise Exception("UPLOADED CATALOGS PRESENT IN catalogs/uploaded. TEST STOPPING. Please remove the user-defined catalogs before continuing.")
            else:
                os.rmdir("catalogs/uploaded")

        trigger = ""
        te([], True)
        params["rm_cats"] = None
        os.mkdir("catalogs/uploaded")
        open("catalogs/uploaded/mock.fits", 'w').close()
        te(["mock"], True)
        os.remove("catalogs/uploaded/mock.fits")


    def test_update_radio_image(self):

        trigger = ""
        params = dict(
            btn = None,
            cont = None,
            bright = None,
            selected_source = None,
            current_image = None,
            current_target = None,
            survey = "NVSS",
            pixels = None,
            target = None,
            autoFocus = None,
            fov = 1,
       )
        ctx = copy_context()

        def run_callback():
            context_value.set(AttributeDict(**{"triggered_inputs":[{"prop_id":trigger}]}))
            return map.update_radio_image(**params)
        # -------------------------------------------------------------------------------
        
        Image, Target = ctx.run(run_callback)
        self.assertTrue(isinstance(Image, html.Img))
        self.assertEqual(Target, '05h35m16.8s -05d23m24s')

        params["current_image"] = "mock"
        trigger = "contrast.value"
        Image, _ = ctx.run(run_callback)
        self.assertTrue(isinstance(Image, html.Img))

        params["autoFocus"] = True
        params["survey"] = "VLA FIRST (1.4 GHz)"
        params["target"] = {"ra": 0, "dec": 90}
        trigger = "get_image.nclicks"
        _, Target = ctx.run(run_callback)
        self.assertEqual(Target, "Try Another Survey")

        trigger = "x.x"
        self.assertTupleEqual(ctx.run(run_callback), ("mock", None))

        params["selected_source"] = {"ra": 0, "dec": 90}
        _, Target = ctx.run(run_callback)
        self.assertEqual(Target, "Try Another Survey")


    def test_select_all_catalogs(self):

        params = {"btn": None}
        ctx = copy_context()
        def run_callback():
            context_value.set(AttributeDict(**{"triggered_inputs":[{"prop_id":""}]}))
            return map.select_all_catalogs(**params)
        # -------------------------------------------------------------------------------

        self.assertListEqual(ctx.run(run_callback), [])
        params["btn"] = 1
        self.assertListEqual(ctx.run(run_callback), ['FR0', 'FRI', 'FRII', 'LRG', 'MiraBest', 'Proctor', 'unLRG'])

    
    def test_select_all_types(self):

        params = {"btn": None}
        ctx = copy_context()
        def run_callback():
            context_value.set(AttributeDict(**{"triggered_inputs":[{"prop_id":""}]}))
            return map.select_all_types(**params)
        # -------------------------------------------------------------------------------

        self.assertListEqual(ctx.run(run_callback), [])
        params["btn"] = 1
        self.assertListEqual(ctx.run(run_callback), ['BENT', 'COMPACT', 'FRI', 'FRII', 'OTHER', 'RING', 'X'])


    def test_update_layers_and_cols(self):

        ctx = copy_context()
        trigger = ""
        params = dict(
            all_cat_btn = None,
            cats = None,
            typs = None,
            upload = None,
            delete_btn = None,
            cats_rm = ["LeroCat"],
            layers = None,
            rad = "Right Ascension",
            decd = "Declination",
            typ = "Morphologies",
            content = 'data:text/csv;base64,UmlnaHQgQXNjZW5zaW9uLERlY2xpbmF0aW9uLE1vcnBob2xvZ2llcwoxMDAsMCxGUkkKMTAwLDIsRlJJSQoxMDAsNCxDT01QQUNUCjEwMCw2LEJFTlQKMTAwLDgsWAoxMDAsMTAsUklORwoxMDAsMTIsT1RIRVIKMTAwLDE0LEZSSQoxMDAsMTYsRlJJSQoxMDAsMTgsRlJJCjEwMCwyMCxGUklJCjEwMCwyMixDT01QQUNUCjEwMCwyNCxCRU5UCjEwMCwyNixYCjEwMCwyOCxSSU5HCjEwMCwzMCxPVEhFUgoxMDAsMzIsRlJJCjEwMCwzNCxGUklJCjEwMCwzNixGUkkKMTAwLDM4LEZSSUkKMTAwLDQwLENPTVBBQ1QKMTAwLDQyLEJFTlQKMTAwLDQ0LFgKMTAwLDQ2LFJJTkcKMTAwLDQ4LE9USEVSCjEwMCw1MCxGUkkKMTAwLDUyLEZSSUkKMTAwLDU0LEZSSQoxMDAsNTYsRlJJSQoxMDAsNTgsQ09NUEFDVAoxMDAsNjAsQkVOVAoxMDAsNjIsWAoxMDAsNjQsUklORwoxMDAsNjYsT1RIRVIKMTAwLDY4LEZSSQoxMDAsNzAsRlJJSQoxMDAsNzIsRlJJCjEwMCw3NCxGUklJCjEwMCw3NixDT01QQUNUCjEwMCw3OCxCRU5UCjEwMCw4MCxYCjEwMCw4MixSSU5HCjEwMCw4NCxPVEhFUgo=',
            filename = "LeroCat"
        )

        def run_callback():
            context_value.set(AttributeDict(**{"triggered_inputs":[{"prop_id":trigger}]}))
            return map.update_layers(**params)
        
        def run_callback2():
            context_value.set(AttributeDict(**{"triggered_inputs":[{"prop_id":trigger}]}))
            return map.new_catalog_columns(**params)
        
        def te(*result):
            self.assertTupleEqual(ctx.run(run_callback), result)

        def te2(*result):
            self.assertTupleEqual(ctx.run(run_callback2), result)
        # -------------------------------------------------------------------------------

        trigger = "upload-button.nclicks"
        te(None, ['FR0', 'FRI', 'FRII', 'LRG', 'MiraBest', 'Proctor', 'unLRG'])

        params["upload"] = True
        te(None, ['FR0', 'FRI', 'FRII', 'LRG', 'MiraBest', 'Proctor', 'unLRG'])

        params['filename'] += ".csv"
        layers, options = ctx.run(run_callback)
        self.assertNotEqual(len(layers), 0)

        #------------------------------------------------
        p = params.copy()
        trigger = ""
        params = dict(
            contents = None,
            filename = "LeroCat",
            upload = None,
            date = 1692863715.14,
            is_hidden = False
        )

        te2([], [], [], [], True, None)
        params["contents"] = "x,x"
        self.assertIsNone(ctx.run(run_callback2))

        params["contents"] = p["content"]
        te2([], [], [], [], True, None)

        params["filename"] += ".csv"
        trigger = "upload-button.n_clicks"
        te2([], [], [], [], True, None)

        params["is_hidden"] = True
        options1, op2, op3, _, boolean, _ = ctx.run(run_callback2)
        self.assertListEqual(options1,op2,op3)
        self.assertFalse(boolean)
        #------------------------------------------------

        params = p
        trigger = "upload-button.nclicks"
        
        params["content"] = "x,x"
        te(None, ['FR0', 'FRI', 'FRII', 'LRG', 'LeroCat', 'MiraBest', 'Proctor', 'unLRG'])

        trigger = "all-catalogs-btn.nclicks"
        params["all_cat_btn"] = 1
        params['layers'] = layers
        layers, _ = ctx.run(run_callback)
        self.assertFalse(layers[0]["options"]["show"])
        params['typs'] = ['BENT', 'COMPACT', 'FRI', 'FRII', 'OTHER', 'RING', 'X']
        layers, _ = ctx.run(run_callback)
        self.assertTrue(layers[0]["options"]["show"])

        params['all_cat_btn'] = 0
        params['layers'] = [] 
        te([], ['FR0', 'FRI', 'FRII', 'LRG', 'LeroCat', 'MiraBest', 'Proctor', 'unLRG'])

        trigger = "delete_button.nclicks"
        params['layers'] = layers
        layers, _ = ctx.run(run_callback)
        self.assertFalse(layers[0]["options"]["show"])


    def test_disable_pixel(self):
        p = {"auto":None}
        ctx = copy_context()
        def run_callback():
            context_value.set(AttributeDict(**{"triggered_inputs":[{"prop_id":""}]}))
            return map.disable_pixel(**p)
        
        self.assertFalse(ctx.run(run_callback))
        p["auto"] = True
        self.assertTrue(ctx.run(run_callback))