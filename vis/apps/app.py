import asyncio
import os

import spade
from spade import wait_until_finished
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour

import anova
import algorithm
import data_split
import messages
import convergence
import inference
import network
import download
from settings import page

from config import Config

config = Config()


async def SETTINGS(request):
    return {"figures": anova.generate(config=config, download=False)}


async def ANOVA(request):
    return {"figures": anova.generate(config=config, download=False)}


async def ALGORITHM(request):
    return {"figures": algorithm.generate(config=config, download=False)}


async def DATA_SPLIT(request):
    return {"figures": data_split.generate(config=config, download=False)}


async def MESSAGES(request):
    return {"figures": messages.generate(config=config, download=False)}


async def CONVERGENCE(request):
    return {"figures": convergence.generate(config=config, download=False)}


async def INFERENCE(request):
    return {"figures": inference.generate(config=config, download=False)}


async def NETWORK(request):
    return {"figures": network.generate(config=config, download=False)}


async def DOWNLOAD(request):
    return {"figures": download.download(config=config)}


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

    dummy.web.add_menu_entry("Settings", "/manager/plots/settings", "fa fa-sliders")
    dummy.web.add_menu_entry("Anova", "/manager/plots/anova", "fa fa-bolt")
    dummy.web.add_menu_entry("Algorithm", "/manager/plots/algorithm", "fa fa-rocket")
    dummy.web.add_menu_entry("Data Split", "/manager/plots/data_split", "fa fa-bomb")
    dummy.web.add_menu_entry("Messages", "/manager/plots/messages", "fa fa-comments")
    dummy.web.add_menu_entry(
        "Convergence", "/manager/plots/convergence", "fa fa-map-pin"
    )
    dummy.web.add_menu_entry("Inference", "/manager/plots/inference", "fa fa-forward")
    dummy.web.add_menu_entry("Network", "/manager/plots/network", "fa fa-sitemap")
    dummy.web.add_menu_entry("Download", "/manager/plots/download", "fa fa-arrow-down")

    dummy.web.add_get(
        "/manager/plots/settings",
        ANOVA,
        template="plots.html",
    )
    dummy.web.add_get(
        "/manager/plots/anova",
        ANOVA,
        template="plots.html",
    )
    dummy.web.add_get(
        "/manager/plots/algorithm",
        ALGORITHM,
        template="template2.html",
    )
    dummy.web.add_get(
        "/manager/plots/data_split",
        DATA_SPLIT,
        template="template2.html",
    )
    dummy.web.add_get(
        "/manager/plots/messages",
        MESSAGES,
        template="template2.html",
    )
    dummy.web.add_get(
        "/manager/plots/convergence",
        CONVERGENCE,
        template="template2.html",
    )
    dummy.web.add_get(
        "/manager/plots/inference",
        INFERENCE,
        template="template2.html",
    )
    dummy.web.add_get(
        "/manager/plots/network",
        NETWORK,
        template="template2.html",
    )
    dummy.web.add_get(
        "/manager/plots/download",
        DOWNLOAD,
        template="template2.html",
    )
    await dummy.start(auto_register=True)
    dummy.web.start(hostname="localhost", port="10000")
    print(dummy.web.menu_entries)
    print("Wait until user interrupts with ctrl+C")
    await wait_until_finished(dummy)


if __name__ == "__main__":
    spade.run(main())
