"""
draw.py - Draw a digit with the mouse and turn it into the network's input.

preprocess() reproduces MNIST's preprocessing (docs/spec.md, sections 2 and 9):
crop to the bounding box, scale the longest side to 20 px, put the center of
mass at (14, 14) in a 28x28 frame, binarize with the training threshold.
The result is the 784-value 0/1 vector the Z80 expects (pixel i = 28*row + col).

DrawingApp is independent of the Z80: host/demo.py passes a callback that
sends the image to the board and returns the digit. The callback runs in a
background thread (Worker), so the window keeps redrawing while the Z80 works;
an optional side panel (host/debug_panel.py) uses the same thread.
With on_stop, a slow classification can be stopped: its answer is dropped,
and a digit asked for meanwhile runs next.

Run `python host/draw.py` to check the preprocessing by eye in the terminal.
"""

import queue
import threading
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

CANVAS_SIZE = 280          # drawing area, 10x the MNIST frame
BRUSH_RADIUS = 10          # px on the canvas
INK_THRESHOLD = 30         # pixels above this count for the bounding box
BOX_SIZE = 20              # longest side of the digit after scaling (MNIST)
FRAME_SIZE = 28            # MNIST frame
FRAME_CENTER = 14          # target (row, col) of the center of mass
BINARY_THRESHOLD = 127     # ToTensor() then x > 0.5  <=>  byte >= 128
PREVIEW_SCALE = 8          # 28 * 8 = 224 px preview
POLL_MS = 20               # how often Tk checks for finished background jobs


def preprocess(img):
    """Turn a 280x280 "L" drawing (white on black) into the network input.

    Returns a (784,) uint8 array of 0/1 values, row by row, or None if
    nothing was drawn.
    """
    pixels = np.asarray(img)
    rows, cols = np.nonzero(pixels > INK_THRESHOLD)
    if rows.size == 0:
        return None
    crop = img.crop((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1))

    scale = BOX_SIZE / max(crop.size)
    size = tuple(max(1, round(side * scale)) for side in crop.size)
    small = crop.resize(size, Image.LANCZOS)

    center = center_of_mass(np.asarray(small, dtype=np.float64))
    if center is None:
        return None  # faint specks that vanished when scaled down
    row, col = center

    frame = Image.new("L", (FRAME_SIZE, FRAME_SIZE), 0)
    # paste() clips whatever falls outside the frame
    frame.paste(small, (round(FRAME_CENTER - col), round(FRAME_CENTER - row)))

    return (np.asarray(frame) > BINARY_THRESHOLD).astype(np.uint8).reshape(-1)


def center_of_mass(pixels):
    """Intensity-weighted (row, col) of a 2-D array, or None if it is all zero."""
    total = pixels.sum()
    if total == 0:
        return None
    rows, cols = np.indices(pixels.shape)
    return (rows * pixels).sum() / total, (cols * pixels).sum() / total


def to_ascii(x):
    """The 28x28 image as 28 lines of '#' (pixel 1) and '.' (pixel 0)."""
    grid = np.asarray(x).reshape(FRAME_SIZE, FRAME_SIZE)
    return "\n".join("".join("#" if p else "." for p in row) for row in grid)


def preview_image(x):
    """A (784,) 0/1 vector as a 224x224 PhotoImage, scaled up with NEAREST."""
    img = Image.fromarray(np.asarray(x, dtype=np.uint8).reshape(FRAME_SIZE, FRAME_SIZE) * 255)
    side = FRAME_SIZE * PREVIEW_SCALE
    return ImageTk.PhotoImage(img.resize((side, side), Image.NEAREST))


class Worker:
    """Runs jobs one at a time in a background thread.

    submit(fn, done): fn() runs in the thread; done(result, error) is then
    called in Tk's thread, where widgets may be touched. One thread means the
    Z80 sees one command at a time, whoever sends it.
    """

    def __init__(self, root):
        self.root = root
        self.jobs = queue.Queue()
        self.results = queue.Queue()
        self.pending = 0
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, fn, done):
        self.pending += 1
        self.jobs.put((fn, done))
        if self.pending == 1:
            self.root.after(POLL_MS, self._poll)

    def _run(self):
        while True:
            fn, done = self.jobs.get()
            try:
                self.results.put((done, fn(), None))
            except Exception as error:      # shown by done(), the thread keeps going
                self.results.put((done, None, error))

    def _poll(self):
        while not self.results.empty():
            done, result, error = self.results.get()
            self.pending -= 1
            done(result, error)
        if self.pending:
            self.root.after(POLL_MS, self._poll)


class DrawingApp(tk.Tk):
    """Window with a drawing canvas, a preview of the network input and the result.

    on_submit(x) -> str, if given, receives the preprocessed (784,) vector on
    "Recognize" and its return value is shown in the result label. It runs in
    the worker thread.
    side_panel(app) -> tk widget, if given, is placed on the right. It can run
    its own jobs with app.run_in_background, and its refresh() is called after
    each result.
    on_stop(), if given, adds Stop: it is called in Tk's thread and must make
    the running on_submit return soon. Its answer is dropped, then
    on_stopped() is called, before any other job starts. Recognize while the
    Z80 works stops it and runs the new drawing next; Clear stops it too.
    """

    def __init__(self, on_submit=None, side_panel=None, on_stop=None, on_stopped=None):
        super().__init__()
        self.title("Draw a digit")
        self.on_submit = on_submit
        self.on_stop = on_stop
        self.on_stopped = on_stopped
        self.busy = False
        self.stopping = False
        self.next_x = None              # drawing to run once the stopped one returns
        self.last_point = None
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pen = ImageDraw.Draw(self.image)
        self.worker = Worker(self)
        self._build_widgets()
        self.panel = None
        if side_panel is not None:
            self.panel = side_panel(self)
            self.panel.grid(row=0, column=2, rowspan=4, sticky="n", padx=10, pady=10)
        self.clear()

    def run_in_background(self, fn, done):
        self.worker.submit(fn, done)

    def _build_widgets(self):
        """Create and lay out the widgets, bind mouse and keyboard."""
        self.canvas = tk.Canvas(self, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", highlightthickness=0, cursor="pencil")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.preview = tk.Label(self, bg="black")
        self.preview.grid(row=0, column=1, padx=10, pady=10)

        buttons = tk.Frame(self)
        buttons.grid(row=1, column=0, columnspan=2)
        self.recognize_button = tk.Button(buttons, text="Recognize", command=self.recognize)
        self.recognize_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = None
        if self.on_stop is not None:
            self.stop_button = tk.Button(buttons, text="Stop", command=self.stop, state=tk.DISABLED)
            self.stop_button.pack(side=tk.LEFT, padx=5)
        tk.Button(buttons, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)

        self.result = tk.Label(self, font=("Helvetica", 48, "bold"))
        self.result.grid(row=2, column=0, columnspan=2)
        self.status = tk.Label(self, font=("Helvetica", 12))
        self.status.grid(row=3, column=0, columnspan=2, pady=(0, 10))

        self.canvas.bind("<Button-1>", self._on_paint)
        self.canvas.bind("<B1-Motion>", self._on_paint)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Return>", lambda _event: self.recognize())
        self.bind("<Escape>", lambda _event: self.clear())

    def _on_paint(self, event):
        """Draw the same stroke on the canvas and on the PIL image."""
        x, y, r = event.x, event.y, BRUSH_RADIUS
        if self.last_point is not None:
            # a line from the previous point avoids gaps when the mouse moves fast
            self.canvas.create_line(*self.last_point, x, y, fill="white", width=2 * r)
            self.pen.line([self.last_point, (x, y)], fill=255, width=2 * r)
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.pen.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self.last_point = (x, y)

    def _on_release(self, _event):
        """End the current stroke."""
        self.last_point = None

    def recognize(self):
        """Preprocess the drawing, show the preview, ask the callback in the background.

        While the Z80 works: stop it and run this drawing next (if it can stop).
        """
        if self.busy and self.on_stop is None:
            return
        x = preprocess(self.image)
        if x is None:
            self.status.config(text="Draw a digit first")
            return
        self._show_preview(x)
        if self.busy:
            self.next_x = x
            self.stop()
            return
        self._submit(x)

    def _submit(self, x):
        self.status.config(text="")
        if self.on_submit is None:
            return
        self.busy = True
        if self.stop_button is None:
            self.recognize_button.config(state=tk.DISABLED)
        else:
            self.stop_button.config(state=tk.NORMAL)
        self.status.config(text="The Z80 is working...")
        self.run_in_background(lambda: self.on_submit(x), self._on_result)

    def stop(self):
        """Ask the running classification to end; _on_result drops its answer."""
        if not self.busy or self.stopping or self.on_stop is None:
            return
        self.stopping = True
        self.stop_button.config(state=tk.DISABLED)
        self.status.config(text="Stopping...")
        self.on_stop()

    def _on_result(self, text, error):
        """Back in Tk's thread: show the answer, then let the panel refresh."""
        self.busy = False
        self.recognize_button.config(state=tk.NORMAL)
        if self.stop_button is not None:
            self.stop_button.config(state=tk.DISABLED)
        if self.stopping:
            self.stopping = False
            if self.on_stopped is not None:
                self.on_stopped()
            self.result.config(text="")
            x, self.next_x = self.next_x, None
            if x is not None:
                self._submit(x)
                return
            self.status.config(text="Stopped: draw another digit")
            if self.panel is not None:
                self.panel.refresh()
            return
        if error is not None:
            self.status.config(text=f"Error: {error}")
            return
        self.status.config(text="")
        self.result.config(text=text)
        if self.panel is not None:
            self.panel.refresh()

    def clear(self):
        """Erase the drawing, the preview and the result; stop the Z80 if it works."""
        self.canvas.delete("all")
        self.pen.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.last_point = None
        self._show_preview(np.zeros(FRAME_SIZE * FRAME_SIZE, dtype=np.uint8))
        self.result.config(text="")
        self.status.config(text="Draw a digit, then press Enter")
        self.next_x = None
        self.stop()

    def _show_preview(self, x):
        """Show x in the preview label (keep a reference so Tk does not drop it)."""
        self.preview_photo = preview_image(x)
        self.preview.config(image=self.preview_photo)


def print_and_ok(x):
    """Callback for manual checks: print the image in the terminal."""
    print(to_ascii(x), end="\n\n")
    return "OK"


if __name__ == "__main__":
    DrawingApp(on_submit=print_and_ok).mainloop()
