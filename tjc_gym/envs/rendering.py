import pyglet
from pyglet.gl import *

key = pyglet.window.key

RAD2DEG = 57.29577951308232


class Viewer(pyglet.window.Window):
    def __init__(self, width, height, *args, **kwargs) -> None:
        super(Viewer, self).__init__(width, height, *args, **kwargs)

        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)

        self.transform = Transform()

        self.geoms = []
        self.labels = []

    def render(self):
        glClearColor(1, 1, 1, 1)

        self.clear()
        self.switch_to()
        self.dispatch_events()

        # draw geoms
        self.transform.enable()
        for geom in self.geoms:
            geom.draw()
        self.transform.disable()

        # draw label - requires special scaling
        translation = self.transform.translation
        scale = self.transform.scale
        glPushMatrix()
        glTranslatef(translation[0], translation[1], 0)
        for label in self.labels:
            pos = label.position
            new_pos = (pos[0] * scale[0], pos[1] * scale[1])
            label.position = new_pos
            label.draw()
            label.position = pos  # reset label position
        glPopMatrix()

        self.flip()

    def reset(self):
        # self.player.delete()
        for geom in self.geoms:
            geom.delete()
            del geom
        for label in self.labels:
            label.delete()
            del label
        self.geoms = []
        self.labels = []

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley))

    def add_circle(self, radius=10, is_foreground=True):
        shape = pyglet.shapes.Circle(
            0, 0, radius=radius, batch=self.batch, group=self.foreground if is_foreground else self.background
        )

        self.geoms.append(shape)

        return shape

    def add_rectangle(self, width, height, is_foreground=True):
        shape = pyglet.shapes.Rectangle(
            0, 0, width, height, batch=self.batch, group=self.foreground if is_foreground else self.background
        )

        self.geoms.append(shape)

        return shape

    def add_line(self, start, end, width=0.05, is_foreground=True):
        shape = pyglet.shapes.Line(
            *start, *end, width=width, batch=self.batch, group=self.foreground if is_foreground else self.background
        )
        self.geoms.append(shape)

        return shape

    def add_label(self, text, x=0.5, y=0.5, size=18, anchor_x="center"):
        label = pyglet.text.Label(
            text,
            font_name="Times New Roman",
            x=x,
            y=y,
            anchor_x=anchor_x,
            anchor_y="center",
            font_size=size,
            color=(255, 255, 255, 255),
        )
        self.labels.append(label)

        return label


class Attribute(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Transform(Attribute):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)) -> None:
        self.set_translation(*translation)
        self.set_scale(*scale)
        self.set_rotation(rotation)

    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))

    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))

    def set_rotation(self, new):
        self.rotation = float(new)

    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0)
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)

    def disable(self):
        glPopMatrix()


class Geom(object):
    def __init__(self) -> None:
        super().__init__()
        self._color = Color(0, 0, 0, 1.0)
        self.attrs = [self._color]

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self._render()
        for attr in self.attrs:
            attr.disable()

    def _render(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b, a=1):
        self._color.rgba_vec = (r, g, b, a)


class FilledPolygon(Geom):
    def __init__(self, v) -> None:
        super().__init__(self)
        self.v = v


class Attr(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass


class Color(Attr):
    def __init__(self, rgba_vec) -> None:
        super().__init__()
        self.rgba_vec = rgba_vec

    def enable(self):
        glColor4f(self.rgba_vec)
