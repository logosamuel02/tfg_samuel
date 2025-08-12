from anova import figures
from bubble_plot_app import bubble_plot
from layer_evolution import layer_evolution_plot
from violin_plot import violin_plot
from network import network_plot

html = """
<span style='font-family:"Arial Black";font-size:35px;'>
<span style='color:#FF0505;'>A</span><span style='color:#FE1304;'>l</span>
<span style='color:#FD2204;'>l</span> <span style='color:#FB3F03;'>i</span>
<span style='color:#FA4E03;'>m</span><span style='color:#F95D03;'>a</span>
<span style='color:#F86B02;'>g</span><span style='color:#F77A02;'>e</span>
<span style='color:#F68902;'>s </span> <span style='color:#F4A601;'>d</span>
<span style='color:#F3B501;'>o</span><span style='color:#F2C401;'>w</span>
<span style='color:#F1D200;'>n</span><span style='color:#F0E100;'>l</span>
<span style='color:#EFF000;'>o</span><span style='color:#EEFF01;'>a</span>
<span style='color:#EFF000;'>d</span><span style='color:#F0E100;'>e</span>
<span style='color:#F1D200;'>d </span> <span style='color:#F3B501;'>s</span>
<span style='color:#F4A601;'>u</span><span style='color:#F59802;'>c</span>
<span style='color:#F68902;'>c</span><span style='color:#F77A02;'>e</span>
<span style='color:#F86B02;'>s</span><span style='color:#F95D03;'>f</span>
<span style='color:#FA4E03;'>u</span><span style='color:#FB3F03;'>l</span>
<span style='color:#FC3104;'>l</span><span style='color:#FD2204;'>y</span>
<span style='color:#FE1304;'>!</span></span>
"""

def download():
    figures(download=True)
    bubble_plot(download=True)
    violin_plot(download=True)
    network_plot(download=True)
    layer_evolution_plot(download=True)
    return html
