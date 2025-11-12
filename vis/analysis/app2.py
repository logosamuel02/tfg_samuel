import asyncio
import os

import spade
from spade import wait_until_finished
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour

import visualization

from visualization import anova
from visualization import algorithm
from visualization import data_split
from visualization import messages
from visualization import convergence
from visualization import inference
from visualization import network

from export import Config, get_figure, load_parameters

config = Config()


async def ANOVA(request):
    return


async def ARGS_TUCKEY(request):
    pams = load_parameters()
    print("Arguments sent")
    return pams["anova"]["tuckey"]


async def GET_TUCKEY(request):
    config = Config()
    filename = config.plots["anova"]["tuckey"]["layout"]["title_text"]
    filename = rf"figures/anova/{filename.replace(' ', '_')}.json"
    return get_figure(filename)


async def GEN_TUCKEY(request):
    form = await request.post()
    form = dict(form)
    fig = anova.create_tuckey_test(**form)
    print("Figure generated sent")
    return fig.layout.title.text, fig.to_html(full_html=False)


async def ARGS_ANOVA_TABLE(request):
    pams = load_parameters()
    print("Arguments sent")
    return pams["anova"]["anova"]


async def GET_ANOVA_TABLE(request):
    config = Config()
    filename = config.plots["anova"]["anova"]["layout"]["title_text"]
    filename = rf"figures/anova/{filename.replace(' ', '_')}.json"
    return get_figure(filename)


async def GEN_ANOVA_TABLE(request):
    form = await request.post()
    form = dict(form)
    fig = anova.create_anova(**form)
    print("Figure generated sent")
    return fig.layout.title.text, fig.to_html(full_html=False)


async def ALGORITHM(request):
    return {"figures": algorithm.generate(config=config, action="return")}


async def DATA_SPLIT(request):
    return {"figures": data_split.generate(config=config, action="return")}


async def MESSAGES(request):
    return {"figures": messages.generate(config=config, action="return")}


async def CONVERGENCE(request):
    return {"figures": convergence.generate(config=config, action="return")}


async def INFERENCE(request):
    return {"figures": inference.generate(config=config, action="return")}


async def NETWORK(request):
    return {"figures": network.generate(config=config, action="return")}


class DummyAgent(Agent):
    class MyBehav(CyclicBehaviour):
        async def on_start(self):
            print("Starting behaviour . . .")
            self.counter = 0

        async def run(self):
            await asyncio.sleep(1)

    async def setup(self):
        print("Agent starting . . .")
        b = self.MyBehav()
        self.add_behaviour(b)
        self.add_behaviour(b)


async def main():
    folder = "images"
    isExist = os.path.exists("images")
    if not isExist:
        os.makedirs(folder)
    dummy = DummyAgent("dummy@localhost", "your_password")
    print("DummyAgent started. Check its console to see the output.")

    dummy.web.add_menu_entry("Anova", "/manager/plots/anova", "fa fa-bolt")

    dummy.web.app.router.add_static(
        "/manager/plots/rfLogo",
        r"C:\Users\samue\OneDrive\Escritorio\Tareas UNI\tfg\tfg_samuel\vis\analysis\rfLogo",
    )

    dummy.web.app.router.add_static(
        "/manager/plots/web",
        r"C:\Users\samue\OneDrive\Escritorio\Tareas UNI\tfg\tfg_samuel\vis\analysis\web",
    )

    dummy.web.add_get(
        "/manager/plots/anova",
        ANOVA,
        template="web/anova.html",
    )

    dummy.web.add_get(
        "/manager/plots/anova/get_tuckey",
        GET_TUCKEY,
        template=None,
    )
    dummy.web.add_get(
        "/manager/plots/anova/args_tuckey",
        ARGS_TUCKEY,
        template=None,
    )
    dummy.web.add_post(
        "/manager/plots/anova/gen_tuckey",
        GEN_TUCKEY,
        template=None,
    )

    dummy.web.add_get(
        "/manager/plots/anova/get_anova_table",
        GET_ANOVA_TABLE,
        template=None,
    )
    dummy.web.add_get(
        "/manager/plots/anova/args_anova_table",
        ARGS_ANOVA_TABLE,
        template=None,
    )
    dummy.web.add_post(
        "/manager/plots/anova/gen_anova_table",
        GEN_ANOVA_TABLE,
        template=None,
    )

    await dummy.start(auto_register=True)
    dummy.web.start(hostname="localhost", port="10000")
    print(dummy.web.menu_entries)
    print("Wait until user interrupts with ctrl+C")
    await wait_until_finished(dummy)


if __name__ == "__main__":
    spade.run(main())
