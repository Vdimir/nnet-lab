from PIL import Image, ImageDraw, ImageChops
import PIL
from tkinter import *
import pickle
import matplotlib.pyplot as plt

import numpy as np

"""paint.py: not exactly a paint program.. just a smooth line drawing demo."""
width = 100
height = 100
image1 = PIL.Image.new("RGB", (width, height), (0,)*3)
draw = ImageDraw.Draw(image1)

b1 = "up"
xold, yold = None, None

with open('data.pkl', 'rb') as f:
    model = pickle.load(f)

pred = -1
pix = None

def trim(im):
    bg = PIL.Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()

    if bbox:
        return im.crop((bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5))


def main():
    root = Tk()
    drawing_area = Canvas(root, width=width, height=height)
    drawing_area.pack()
    drawing_area.bind("<Motion>", motion)
    drawing_area.bind("<ButtonPress-1>", b1down)
    drawing_area.bind("<ButtonRelease-1>", b1up)
    drawing_area.bind("<ButtonRelease-2>", b2up)
    drawing_area.bind("<ButtonRelease-3>", b3up)

    root.geometry('%dx%d' % (200,200))
    root.mainloop()

def b1down(event):
    global b1
    b1 = "down"

def b1up(event):
    global b1, xold, yold
    b1 = "up"
    xold = None
    yold = None

def clear(event):
    global image1, draw
    image1 = PIL.Image.new("RGB", (width, height), (0,) * 3)
    draw = ImageDraw.Draw(image1)
    event.widget.delete("all")
    update_status(event)

def update_status(event):
    event.widget.winfo_toplevel().title("%d" % (pred,))

def b2up(event):
    clear(event)

def b3up(event):
    clear(event)


fill_with = 7
def motion(event):
    if b1 == "down":
        global xold, yold, pix, pred
        if xold is not None and yold is not None:
            p = fill_with//2
            event.widget.create_line(xold, yold, event.x, event.y, smooth=True, width=fill_with)
            event.widget.create_oval(event.x-p, event.y-p, event.x + p, event.y + p, fill="black")

            draw.line([xold, yold, event.x, event.y], 255, width=fill_with)
            draw.ellipse([event.x-p, event.y-p, event.x + p, event.y + p], 255)
            scaled_img = (image1).resize((28, 28), PIL.Image.ANTIALIAS)

            pix = np.array(scaled_img.getdata()).reshape(scaled_img.size[0], scaled_img.size[1], 3)
            pix = pix[:,:,0].astype(np.float) / 255.0
            p = model.predict(pix.reshape(1, 28 * 28))
            pred = p[0]
            update_status(event)

        xold = event.x
        yold = event.y

if __name__ == "__main__":
    main()
    plt.imshow(pix, cmap = 'gray')
    plt.show()
