from jinja2 import Environment, FileSystemLoader


def hello_world():
    return "hello world from within the function"


def multiply(x, y):
    return str(x * y)


func_dict = {
    "hello_world": hello_world,
    "multiply": multiply,
}


def render(template):
    env = Environment(loader=FileSystemLoader(""))
    jinja_template = env.get_template(template)
    jinja_template.globals.update(func_dict)
    template_string = jinja_template.render()
    return template_string


if __name__ == "__main__":
    # print(render(template="prueba.html"))
    from pathlib import Path

    file = Path(__file__).name
    print(file)
