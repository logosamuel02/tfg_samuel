import asyncio


import spade
from spade import wait_until_finished
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour

import plotly.express as px


def plot_fig():
    df = px.data.gapminder()
    fig = px.scatter(
        df,
        x="gdpPercap",
        y="lifeExp",
        animation_frame="year",
        animation_group="country",
        size="pop",
        color="continent",
        hover_name="country",
        log_x=True,
        size_max=55,
        range_x=[100, 100000],
        range_y=[25, 90],
    )
    return fig.to_html(full_html=False)


async def hello_controller(request):
    return {"fig": plot_fig()}


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
        # self.web.add_menu_entry("My entry", "/home", "fa fa-user")
        # self.web.add_get("/home", hello_controller, template="template.html")


async def main():
    dummy = DummyAgent("dummy@localhost", "your_password")
    print("DummyAgent started. Check its console to see the output.")
    dummy.web.add_menu_entry("My entry", "/home", "fa fa-user")
    dummy.web.add_get("/home", hello_controller, template="template.html")
    await dummy.start(auto_register=True)
    dummy.web.start(hostname="localhost", port="10000")
    print(dummy.web.menu_entries)
    print("Wait until user interrupts with ctrl+C")
    await wait_until_finished(dummy)


if __name__ == "__main__":
    spade.run(main())
