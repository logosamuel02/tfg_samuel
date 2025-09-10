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

from dotenv import load_dotenv
from config import Config

load_dotenv()
config = Config()

async def ANOVA(request):
    return {"lista": anova.generate(config=config, download=False)}

async def ALGORITHM(request):
    return {"lista": algorithm.generate(config=config, download=False)}

async def DATA_SPLIT(request):
    return {"lista": data_split.generate(config=config, download=False)}

async def MESSAGES(request):
    return {"lista": messages.generate(config=config, download=False)}

async def CONVERGENCE(request):
    return {"lista": convergence.generate(config=config, download=False)}

async def INFERENCE(request):
    return {"lista": inference.generate(config=config, download=False)}

async def NETWORK(request):
    return {"lista": network.generate(config=config, download=False)}

async def DOWNLOAD(request):
    return {"lista": download.download(config=config)}


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
    
    dummy.web.add_menu_entry("Anova", "/spade/anova", "fa fa-bolt")
    dummy.web.add_menu_entry("Algorithm", "/spade/algorithm", "fa fa-bomb")
    dummy.web.add_menu_entry("Data Split", "/spade/data_split", "fa fa-bomb")
    dummy.web.add_menu_entry("Messages", "/spade/messages", "fa fa-bomb")
    dummy.web.add_menu_entry("Convergence", "/spade/convergence", "fa fa-bomb")
    dummy.web.add_menu_entry("Inference", "/spade/inference", "fa fa-bomb")
    dummy.web.add_menu_entry("Network", "/spade/network", "fa fa-bomb")
    dummy.web.add_menu_entry("Download", "/spade/download", "fa fa-bomb")
    
    dummy.web.add_get(
        "/spade/anova",
        ANOVA,
        template="template2.html",
    )
    dummy.web.add_get(
        "/spade/algorithm",
        ALGORITHM,
        template="template2.html",
    )
    dummy.web.add_get(
        "/spade/data_split",
        DATA_SPLIT,
        template="template2.html",
    )
    dummy.web.add_get(
        "/spade/messages",
        MESSAGES,
        template="template2.html",
    )
    dummy.web.add_get(
        "/spade/convergence",
        CONVERGENCE,
        template="template2.html",
    )
    dummy.web.add_get(
        "/spade/inference",
        INFERENCE,
        template="template2.html",
    )
    dummy.web.add_get(
        "/spade/network",
        NETWORK,
        template="template2.html",
    )
    dummy.web.add_get(
        "/spade/download",
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
