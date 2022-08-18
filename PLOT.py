''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput
from bokeh.plotting import figure, show
import numpy as np

# Set up data



def plot_slider(x, y, calculate_y):
    source = ColumnDataSource(data=dict(x=x, y=y))
    TOOLTIPS = [
        ("(x,y)", "($x, $y)"),
    ]
    # Set up plot
    plot = figure(height=400, width=400, title="Analyze function",
                  tools="crosshair,pan,reset,save,wheel_zoom",
                  x_range=[0., 5.], y_range=[0., 5.],tooltips=TOOLTIPS)

    plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)

    # Set up widgets
    text = TextInput(title="title", value='Analyzed function')
    miu = Slider(title="miu", value=0.0, start=0.0, end=2.5, step=0.01)

    # amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
    # phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
    # freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)

    # Set up callbacks
    def update_title(attrname, old, new):
        plot.title.text = text.value

    text.on_change('value', update_title)

    def update_data(attrname, old, new):
        # Get the current slider values
        a = miu.value

        # Generate the new curve
        # x = np.linspace(0, 4*np.pi, N)
        y = calculate_y(h, miu=a)

        source.data = dict(x=h, y=y)

    for w in [miu]:
        w.on_change('value', update_data)

    # Set up layouts and add to document
    inputs = column(text, miu)

    curdoc().add_root(row(inputs, plot, width=800))
    curdoc().title = "Sliders"

# show(plot)


#main
N = 200
h = np.linspace(-11, 11, N)


def get_sigma_w(h, miu, n=None):
    return h / (np.abs(h) ** 2 + miu)


y = get_sigma_w(h, miu=0, n=None)

plot_slider(x=h,y=y,calculate_y=get_sigma_w)