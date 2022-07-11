"""
This app allows users to load Tissuecyte files and create an array that is a colored volume.
"""

import os
import sys
import numpy as np
from glob import glob
from pathlib import Path
from zipfile import ZipFile
import SimpleITK as sitk
# from functools import partial
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QFormLayout, QFileDialog, QSlider, QLabel, QLineEdit
# import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QIcon, QKeyEvent, QImage, QPixmap, QColor, QIntValidator
from PyQt5.QtCore import pyqtSlot, Qt
from PIL import Image, ImageQt


DEFAULT_SLICE = 200
DEFAULT_COLOR_VALUES = [[0, 3000], [0, 3000], [0, 1000]]

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Tissuecyte Colored Volume'
        self.left = 500
        self.top = 100
        self.width = 1200
        self.height = 800
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        grid = QGridLayout()

        self.image = QLabel()
        self.image.setObjectName("image")
        # self.image.mousePressEvent = self.clickedOnImage
        im8 = Image.fromarray(np.ones((800,800),dtype='uint8')*255)
        imQt = QImage(ImageQt.ImageQt(im8))
        imQt.convertToFormat(QImage.Format_ARGB32)
        self.image.setPixmap(QPixmap.fromImage(imQt))
        grid.addWidget(self.image, 0, 0)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(528)
        self.slider.setValue(DEFAULT_SLICE)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(50)
        self.slider.valueChanged.connect(self.sliderMoved)
        grid.addWidget(self.slider, 1, 0)
        self.slider_values = DEFAULT_SLICE

        subgrid0 = QFormLayout()
        ## RED intensities ##
        self.red_low = QLineEdit()
        self.red_low.setValidator(QIntValidator())
        self.red_low.setMaxLength(10)
        self.red_low.setText(str(DEFAULT_COLOR_VALUES[0][0]))
        self.red_high = QLineEdit()
        self.red_high.setValidator(QIntValidator())
        self.red_high.setMaxLength(10)
        self.red_high.setText(str(DEFAULT_COLOR_VALUES[0][1]))
        ## GREEN intensities ##
        self.green_low = QLineEdit()
        self.green_low.setValidator(QIntValidator())
        self.green_low.setMaxLength(10)
        self.green_low.setText(str(DEFAULT_COLOR_VALUES[1][0]))
        self.green_high = QLineEdit()
        self.green_high.setValidator(QIntValidator())
        self.green_high.setMaxLength(10)
        self.green_high.setText(str(DEFAULT_COLOR_VALUES[1][1]))
        ## BLUE intensities ##
        self.blue_low = QLineEdit()
        self.blue_low.setValidator(QIntValidator())
        self.blue_low.setMaxLength(10)
        self.blue_low.setText(str(DEFAULT_COLOR_VALUES[2][0]))
        self.blue_high = QLineEdit()
        self.blue_high.setValidator(QIntValidator())
        self.blue_high.setMaxLength(10)
        self.blue_high.setText(str(DEFAULT_COLOR_VALUES[2][1]))
        ## Update intensities button ##
        update_color_button = QPushButton('Update', self)
        update_color_button.setToolTip('Update the color volume')
        update_color_button.clicked.connect(self.updateVolume)

        ## Add rows ##
        subgrid0.addRow(QLabel('Set the color intensity ranges:'))
        subgrid0.addRow('Red min', self.red_low)
        subgrid0.addRow('Red max', self.red_high)
        subgrid0.addRow('Green min', self.green_low)
        subgrid0.addRow('Green max', self.green_high)
        subgrid0.addRow('Blue min', self.blue_low)
        subgrid0.addRow('Blue max', self.blue_high)
        subgrid0.addRow(update_color_button)
        grid.addLayout(subgrid0, 0, 1)

        ## Add Load and Save buttons ##
        subgrid1 = QGridLayout()
        save_button = QPushButton('Save', self)
        save_button.setToolTip('Save values as CSV')
        save_button.clicked.connect(self.saveData)

        load_button = QPushButton('Load', self)
        load_button.setToolTip('Load volume data')
        load_button.clicked.connect(self.loadData)

        subgrid1.addWidget(save_button, 3, 0)
        subgrid1.addWidget(load_button, 3, 1)

        grid.addLayout(subgrid1, 1, 1)

        self.current_directory = '//allen/programs/'
        self.data_loaded = False
        self.setLayout(grid)
        self.show()

    def sliderMoved(self):
        self.slider_values = self.slider.value()
        self.refreshImage()

    def refreshImage(self):
        if self.data_loaded:
            plane = self.volume[self.slider.value(),:,:,:]
        else:
            plane = np.ones((320, 456, 3),dtype='uint8')*255
        image = plane.copy()
        height, width, channels = image.shape
        bytesPerLine = channels * width
        imQt = QImage(
            image.data, width, height, bytesPerLine, QImage.Format_RGB888
        )
        pxmap = QPixmap.fromImage(imQt).scaledToWidth(self.width).scaledToHeight(self.height)
        self.image.setPixmap(pxmap)
        self.setGeometry(self.left, self.top, self.width, self.height)

    def loadData(self):
        fname, filt = QFileDialog.getOpenFileName(self,
            caption='Select resampled_images zip file',
            directory=self.current_directory,
            filter='*.zip')

        print(fname)

        self.current_directory = Path(fname).parents[2]
        self.output_file = os.path.join(self.current_directory, 'resampled_color_volume.npy')

        ### Unzip resampled images, unless already done ###
        resampled_images = glob(os.path.join(self.current_directory, 'resampled_*.mhd'), recursive=True)
        if len(resampled_images) == 0:
            print('Extracting resampled images...')
            with ZipFile(fname, 'r') as zipObj:
                zipObj.extractall(path=self.current_directory)

        ### Load resampled volumes ###
        print('Loading resampled images...')
        intensity_arrays = {}
        for imcolor in ['red', 'green', 'blue']:
            resamp_image = sitk.ReadImage(os.path.join(self.current_directory, 'resampled_' + imcolor + '.mhd'))
            intensity_arrays[imcolor] = sitk.GetArrayFromImage(resamp_image).T
        self.int_arrays = intensity_arrays
        self.volume = self.getColorVolume()
        self.data_loaded = True
        self.refreshImage()

    def getColorVolume(self, rgb_levels=DEFAULT_COLOR_VALUES):
        level_adjusted_arrays = []
        for colori, int_level in zip(['red', 'green', 'blue'], rgb_levels):
            colarray = np.clip(self.int_arrays[colori], a_min=int_level[0], a_max=int_level[1]) - int_level[0]
            colarray = (colarray * 255. / (int_level[1] - int_level[0])).astype('uint8')
            level_adjusted_arrays.append(colarray)
        return np.stack(level_adjusted_arrays, axis=-1)

    def updateVolume(self):
        self.volume = self.getColorVolume(
            rgb_levels=[
                [int(self.red_low.text()), int(self.red_high.text())],
                [int(self.green_low.text()), int(self.green_high.text())],
                [int(self.blue_low.text()), int(self.blue_high.text())]
            ]
        )
        self.data_loaded = True
        self.refreshImage()

    def saveData(self):
        if self.data_loaded:
            np.save(self.output_file, self.volume)
            print('Colored volume saved as .npy file.')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
