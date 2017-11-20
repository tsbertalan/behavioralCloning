"""Make some modifications to widgets from tk_tools"""
import cmath, tkinter as tk, tk_tools, numpy as np
from tk_tools.images import rotary_scale


root = tk.Tk()

class RotaryScale(tk_tools.RotaryScale):
    """
    Shows a rotary scale, much like a speedometer.
    """
    def __init__(self, 
        parent=root, min_value=0.0, max_value=100.0, size=100, unit='', img_data='',
        needle_color='white', needle_thickness=0, name='', 
        angleDirect=False, **options
        ):
        """
        Initializes the RotaryScale object

        :param parent: tkinter parent frame
        :param max_value: the value corresponding to the maximum
        value on the scale
        :param size: the size in pixels
        :param options: the frame options
        """
        tk_tools.Dial.__init__(self, parent, size=size, **options)

        self.max_value = float(max_value)
        self.min_value = float(min_value)
        self.size = size
        self.unit = unit
        self.needle_color = needle_color
        self.needle_thickness = needle_thickness
        self.angleDirect = angleDirect

        row = 0
        if name != '':
            self.name = tk.Label(self, text=name)
            self.name.grid(row=row)
            row += 1
        self.canvas = tk.Canvas(self, width=self.size, height=self.size)
        self.canvas.grid(row=row)
        row += 1
        self.readout = tk.Label(self, text='-{}'.format(self.unit))
        self.readout.grid(row=row)

        if img_data:
            if img_data == 'emptyGauge':
                from base64 import b64encode
                img_data = b64encode(open('emptyGauge.png', 'rb').read())
            self.image = tk.PhotoImage(data=img_data)
        else:
            self.image = tk.PhotoImage(data=rotary_scale)

        self.image = self.image.subsample(int(200 / self.size),
                                          int(200 / self.size))

        initial_value = max_value / 2.
        self.set_value(initial_value)

        self.grid()

    def set_value(self, number: float, displayValue=None):
        """
        Sets the value of the graphic

        :param number: the number (must be between 0 and 'max_range'
        or the scale will peg the limits
        :return: None
        """
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.image, anchor='nw')

        if not self.angleDirect:
            number = number if number <= self.max_value else self.max_value
            number = self.min_value if number < self.min_value else number

        radius = 0.9 * self.size/2.0
        if self.angleDirect:
            angle_in_radians = number * np.pi / 180
        else:
            angle_in_radians = (2.0 * cmath.pi / 3.0) \
                + number / self.max_value * (5.0 * cmath.pi / 3.0)

        center = cmath.rect(0, 0)
        outer = cmath.rect(radius, angle_in_radians)
        if self.needle_thickness == 0:
            line_width = int(5 * self.size / 200)
            line_width = 1 if line_width < 1 else line_width
        else:
            line_width = self.needle_thickness

        self.canvas.create_line(
            *self.to_absolute(center.real, center.imag),
            *self.to_absolute(outer.real, outer.imag),
            width=line_width,
            fill=self.needle_color
        )

        self.readout['text'] = '{:.4g}{}'.format(
            displayValue if displayValue else number,
            self.unit
        )
