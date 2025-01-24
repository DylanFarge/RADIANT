from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from sys import argv

opt = Options()
opt.add_experimental_option("detach", True)
opt.add_argument("--start-maximized")
driver = webdriver.Edge(options=opt)

driver.get("http://127.0.0.1:8050/")

if len(argv) > 1 and argv[1].lower() in ["analysis", "downloads"]:
    driver.find_element(By.PARTIAL_LINK_TEXT, argv[1].capitalize()).click()

driver.execute_script(
    '''
    javascript: (function() {
        function callback() {
            gremlins.createHorde({
                species: [gremlins.species.clicker(),gremlins.species.toucher(),gremlins.species.formFiller(),gremlins.species.scroller(),gremlins.species.typer()],
                mogwais: [gremlins.mogwais.alert(),gremlins.mogwais.fps(),gremlins.mogwais.gizmo()],
                strategies: [gremlins.strategies.distribution()]
            }).unleash();
        }
        var s = document.createElement("script");
        s.src = "https://unpkg.com/gremlins.js";
        if (s.addEventListener) {
            s.addEventListener("load", callback, false);
        } else if (s.readyState) {
            s.onreadystatechange = callback;
        }
        document.body.appendChild(s);
        })()
    '''
)