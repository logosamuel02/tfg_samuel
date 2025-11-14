import asyncio
import os

import spade
from spade import wait_until_finished
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour

from visualization import plots


from export import Config, get_figure, load_parameters, load_functions

config = Config()


async def ANOVA(request):
    return


async def GET_ARGS(request):
    form = await request.post()
    form = list(form)[0]
    print("args", form)
    pams = load_parameters()
    return pams[form]


async def GET_FIGURE(request):
    form = await request.post()
    form = list(form)[0]
    print("get", form)
    config = Config()
    filename = config.plots[form]["filename"]
    filename = rf"figures/anova/{filename.replace(' ', '_')}.json"
    return get_figure(filename)


async def GEN_FIGURE(request):
    form = await request.post()
    form = dict(form)
    print("gen", form)
    config = Config()
    filename = config.plots[form["name"]]["filename"]
    pams = load_functions()
    gen_func = getattr(plots, pams[form["name"]])
    try:
        fig = gen_func(**form)
        return filename, fig.to_html(full_html=False)
    except Exception as e:
        print(e)
        return filename, "Some of the attributes selected is not valid"


# async def ALGORITHM(request):
#     return {"figures": algorithm.generate(config=config, action="return")}


# async def DATA_SPLIT(request):
#     return {"figures": data_split.generate(config=config, action="return")}


# async def MESSAGES(request):
#     return {"figures": messages.generate(config=config, action="return")}


# async def CONVERGENCE(request):
#     return {"figures": convergence.generate(config=config, action="return")}


# async def INFERENCE(request):
#     return {"figures": inference.generate(config=config, action="return")}


# async def NETWORK(request):
#     return {"figures": network.generate(config=config, action="return")}


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

    dummy.web.add_post(
        "/manager/plots/anova/get_figure",
        GET_FIGURE,
        template=None,
    )
    dummy.web.add_post(
        "/manager/plots/anova/get_args",
        GET_ARGS,
        template=None,
    )
    dummy.web.add_post(
        "/manager/plots/anova/generate_figure",
        GEN_FIGURE,
        template=None,
    )

    await dummy.start(auto_register=True)
    dummy.web.start(hostname="localhost", port="10000")
    print(dummy.web.menu_entries)
    print("Wait until user interrupts with ctrl+C")
    await wait_until_finished(dummy)


if __name__ == "__main__":
    spade.run(main())
