import os
import numpy as np
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui.widgets import FileEdit
from pathlib import Path
from tifffile import TiffFile, imread, imwrite


class LoadWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Open")
        btn.clicked.connect(self._on_click)

        self.file_edit = FileEdit(label="File: ")

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.file_edit.native)
        self.layout().addWidget(btn)

        self.layers = None

    def _on_click(self):
        path = Path(self.file_edit.value)
        name = path.stem
        self.new_path = os.path.join(path.parents[0], name)
        Path(self.new_path).mkdir(exist_ok=True)

        self.img = TiffFile(path).series[0].asarray()
        lbl = np.zeros_like(self.img[:, 0, ...], dtype=int)

        for i in range(lbl.shape[0]):
            lbl_file = os.path.join(self.new_path, f"{i:02d}_lbl.tif")
            try:
                lbl[i, ...] = imread(lbl_file)
            except FileNotFoundError:
                pass

        self.layers = []
        channel_names = ["red", "green", "phase contrast"]
        channel_colormaps = ["red", "green", "gray"]
        for i in range(len(channel_names)):
            opacity = 0.5 if i > 0 else 1
            tmp_layer = self.viewer.add_image(
                self.img[:, i, ...],
                name=channel_names[i],
                colormap=channel_colormaps[i],
                opacity=opacity,
            )
            self.layers.append(tmp_layer)

        self.lbl_layer = self.viewer.add_labels(lbl)

        self.viewer.bind_key("s")(self.save)
        self.viewer.bind_key("e")(self.select_label)

    def save(self, event=None):
        i = self.viewer.dims.current_step[0]
        lbls = self.lbl_layer.data[i, ...]
        if np.any(lbls) > 0:
            lbl_file = os.path.join(self.new_path, f"{i:02d}_lbl.tif")
            img_file = os.path.join(self.new_path, f"{i:02d}_img.tif")
            imwrite(lbl_file, self.lbl_layer.data[i, ...])
            imwrite(img_file, self.img[i, ...])

    def select_label(self, event=None):
        i = self.viewer.dims.current_step[0]
        self.lbl_layer.selected_label = self.lbl_layer.data[i, ...].max() + 1


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [LoadWidget]
