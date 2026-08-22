class Annotation:
    def __init__(self, x, y, width, height, type):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = type

    def __str__(self):
        return f"(x: {self.x}, y: {self.y})\n(w: {self.width}, h: {self.height})\ntype: {self.type}"