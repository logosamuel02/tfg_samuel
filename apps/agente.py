import asyncio


import spade
from spade import wait_until_finished
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour

from anova import figures
from bubble_plot_app import bubble_plot
from layer_evolution import layer_evolution_plot
from violin_plot import violin_plot
from network import network_plot


async def ANOVA(request):
    return {"lista": figures()}


async def BUBBLE(request):
    return {"fig": bubble_plot()}


async def LAYERS(request):
    return {"lista": layer_evolution_plot()}


async def VIOLIN(request):
    return {"fig": violin_plot()}


async def NETWORK(request):
    return {"fig": network_plot()}


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
    dummy = DummyAgent("dummy@localhost", "your_password")
    print("DummyAgent started. Check its console to see the output.")
    dummy.web.add_menu_entry("Anova", "/spade/anova", "fa fa-bolt")
    dummy.web.add_menu_entry("Bubble", "/spade/bubble_plot", "fa fa-bomb")
    dummy.web.add_menu_entry("Layers", "/spade/layer_plot", "fa fa-bomb")
    dummy.web.add_menu_entry("Violin", "/spade/violin_plot", "fa fa-bomb")
    dummy.web.add_menu_entry("Network", "/spade/network_plot", "fa fa-bomb")
    dummy.web.add_get(
        "/spade/anova",
        ANOVA,
        template="template2.html",
    )
    dummy.web.add_get(
        "/spade/bubble_plot",
        BUBBLE,
        template="template.html",
    )
    dummy.web.add_get(
        "/spade/layer_plot",
        LAYERS,
        template="template2.html",
    )
    dummy.web.add_get(
        "/spade/violin_plot",
        VIOLIN,
        template="template.html",
    )
    dummy.web.add_get(
        "/spade/network_plot",
        NETWORK,
        template="template.html",
    )
    await dummy.start(auto_register=True)
    dummy.web.start(hostname="localhost", port="10000")
    print(dummy.web.menu_entries)
    print("Wait until user interrupts with ctrl+C")
    await wait_until_finished(dummy)


if __name__ == "__main__":
    spade.run(main())
