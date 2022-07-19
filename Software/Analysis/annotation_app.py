"""

This app allows a user to browse through a Tissuecyte volume and mark probe track locations.

"""

import sys
from functools import partial
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QFileDialog, QSlider, QLabel
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QIcon, QKeyEvent, QImage, QPixmap, QColor
from PyQt5.QtCore import pyqtSlot, Qt
import SimpleITK as sitk

from PIL import Image, ImageQt

import numpy as np
import pandas as pd

import os
import time

DEFAULT_SLICE = 200
DEFAULT_VIEW = 0
SCALING_FACTOR = 1.5

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Tissuecyte Annotation'
        self.left = 500
        self.top = 100
        # self.width = int(400*SCALING_FACTOR)
        # self.height = int(400*SCALING_FACTOR)
        self.width = int(456*SCALING_FACTOR) #default = coronal view
        self.height = int(320*SCALING_FACTOR)
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        grid = QGridLayout()

        self.image = QLabel()
        self.image.setObjectName("image")
        self.image.mousePressEvent = self.clickedOnImage
        im8 = Image.fromarray(np.ones((self.height,self.width),dtype='uint8')*255)
        imQt = QImage(ImageQt.ImageQt(im8))
        imQt.convertToFormat(QImage.Format_ARGB32)
        self.image.setPixmap(QPixmap.fromImage(imQt))
        grid.addWidget(self.image, 0, 0)

        self.slider = QSlider(Qt.Horizontal)"""

This app allows a user to browse through a Tissuecyte volume and mark probe track locations.

"""

from sre_constants import SUBPATTERN
import sys
from functools import partial
from unittest.mock import DEFAULT
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QFileDialog, QSlider, QLabel, QFormLayout, QLineEdit, QVBoxLayout, QCheckBox, QRadioButton, QHBoxLayout, QSizePolicy
import PyQt5.QtWidgets as QtWidgets
from PyQt5.QtGui import QIcon, QKeyEvent, QImage, QPixmap, QColor, QIntValidator
from PyQt5.QtCore import pyqtSlot, Qt, QRect
from PyQt5 import QtGui
from qtrangeslider import QRangeSlider
import SimpleITK as sitk
from glob import glob
from pathlib import Path
from zipfile import ZipFile
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PIL import Image, ImageQt
import pyqtgraph as pg

import numpy as np
import pandas as pd

import os
import time

DEFAULT_SLICE = 200
DEFAULT_VIEW = 0
SCALING_FACTOR = 1.5
DEFAULT_COLOR_VALUES = [[0, 3000], [0, 3000], [0, 1000]]

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Tissuecyte Annotation'
        self.left = 500
        self.top = 100
        # self.width = int(400*SCALING_FACTOR)
        # self.height = int(400*SCALING_FACTOR)
        self.width = int(456*SCALING_FACTOR) #default = coronal view
        self.height = int(320*SCALING_FACTOR)

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        rect = QRect(self.left, self.top, self.width, self.height)
        size = rect.size()

        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.left_grid = QVBoxLayout()
        self.main_grid = QHBoxLayout()
        self.right_grid = QVBoxLayout()

        grid = QGridLayout()
        
        self.image = QLabel()
        self.image.setObjectName("image")
        self.image.mousePressEvent = self.clickedOnImage
        im8 = Image.fromarray(np.ones((self.height,self.width),dtype='uint8')*255)
        imQt = QImage(ImageQt.ImageQt(im8))
        imQt.convertToFormat(QImage.Format_ARGB32)
        self.image.setPixmap(QPixmap.fromImage(imQt))
        self.image.setScaledContents(False)
        #grid.addWidget(self.image, 0, 0)
        self.left_grid.addWidget(self.image)


        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(528)
        self.slider.setValue(DEFAULT_SLICE)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(50)
        self.slider.valueChanged.connect(self.sliderMoved)
        #grid.addWidget(self.slider, 1, 0)
        self.left_grid.addWidget(self.slider)

        self.slider_values = [DEFAULT_SLICE, DEFAULT_SLICE, DEFAULT_SLICE]

        subgrid = QGridLayout()

        self.probes = ('A1', 'B1', 'C1', 'D1', 'E1', 'F1',
                       'A2', 'B2', 'C2', 'D2', 'E2', 'F2',
                       'A3', 'B3', 'C3', 'D3', 'E3', 'F3',
                       'A4', 'B4', 'C4', 'D4', 'E4', 'F4')

        self.probe_map = {'Probe A1': 0, 'Probe B1': 1, 'Probe C1': 2, 'Probe D1' : 3,
                          'Probe E1': 4, 'Probe F1': 5,
                          'Probe A2': 6, 'Probe B2': 7, 'Probe C2': 8, 'Probe D2' : 9,
                          'Probe E2': 10, 'Probe F2': 11,
                          'Probe A3': 6, 'Probe B3': 7, 'Probe C3': 8, 'Probe D3' : 9,
                          'Probe E3': 10, 'Probe F3': 11,
                     }

        self.color_map = {'Probe A1': 'darkred', 'Probe B1': 'cadetblue',
                     'Probe C1': 'goldenrod', 'Probe D1' : 'darkgreen',
                     'Probe E1': 'darkblue', 'Probe F1': 'blueviolet',
                     'Probe A2': 'red', 'Probe B2': 'darkturquoise',
                     'Probe C2': 'yellow', 'Probe D2' : 'green',
                     'Probe E2': 'blue', 'Probe F2': 'violet'}

        self.probe_buttons = [QPushButton('Probe ' + i) for i in self.probes]

        self.coronal_button = QPushButton('Coronal', self)
        self.coronal_button.setToolTip('Switch to coronal view')
        self.coronal_button.clicked.connect(self.viewCoronal)

        self.horizontal_button = QPushButton('Horizontal', self)
        self.horizontal_button.setToolTip('Switch to horizontal view')
        self.horizontal_button.clicked.connect(self.viewHorizontal)

        self.sagittal_button = QPushButton('Sagittal', self)
        self.sagittal_button.setToolTip('Switch to sagittal view')
        self.sagittal_button.clicked.connect(self.viewSagittal)

        self.current_view = DEFAULT_VIEW

        subgrid.addWidget(self.coronal_button,2,0,1,2)
        subgrid.addWidget(self.horizontal_button,2,2,1,2)
        subgrid.addWidget(self.sagittal_button,2,4,1,2)

        save_button = QPushButton('Save', self)
        save_button.setToolTip('Save values as CSV')
        save_button.clicked.connect(self.saveData)

        load_button = QPushButton('Load', self)
        load_button.setToolTip('Load volume data')
        load_button.clicked.connect(self.loadData)

        subgrid.addWidget(save_button,3,4)
        subgrid.addWidget(load_button,3,5)

        subgrid_color = QFormLayout()
        subgrid_buttons = QFormLayout()

        self.sc = MplCanvas()

        self.red_button = QCheckBox('Toggle Red')
        self.red_old = None
        self.is_red_checked = False
        self.red_button.clicked.connect(self.toggle_red)

        self.green_button = QCheckBox('Toggle Green')
        self.green_old = None
        self.green_button.clicked.connect(self.toggle_green)
        self.is_green_checked = False

        self.blue_button = QCheckBox('Toggle Blue')
        self.blue_button.clicked.connect(self.toggle_blue)
        self.blue_old = None
        self.is_blue_checked = False

        subgrid_buttons.addWidget(self.red_button)
        subgrid_buttons.addWidget(self.green_button)
        subgrid_buttons.addWidget(self.blue_button)

        subgrid_slider_buttons = QHBoxLayout()

        self.red_slider = QRangeSlider(Qt.Horizontal)
        self.red_slider.setMinimum(DEFAULT_COLOR_VALUES[0][0])
        self.red_slider.setMaximum(DEFAULT_COLOR_VALUES[0][1])
        self.red_slider.setValue((DEFAULT_COLOR_VALUES[0][0], DEFAULT_COLOR_VALUES[0][1]))
        self.red_slider.setTickPosition(QSlider.TicksBelow)
        self.red_slider.setTickInterval(50)
        self.red_slider.valueChanged.connect(self.red_slider_moved)

        self.green_slider = QRangeSlider(Qt.Horizontal)
        self.green_slider.setMinimum(DEFAULT_COLOR_VALUES[1][0])
        self.green_slider.setMaximum(DEFAULT_COLOR_VALUES[1][1])
        self.green_slider.setValue((DEFAULT_COLOR_VALUES[1][0], DEFAULT_COLOR_VALUES[1][1]))
        self.green_slider.setTickPosition(QSlider.TicksBelow)
        self.green_slider.setTickInterval(50)
        self.green_slider.valueChanged.connect(self.green_slider_moved)

        self.blue_slider = QRangeSlider(Qt.Horizontal)
        self.blue_slider.setMinimum(DEFAULT_COLOR_VALUES[2][0])
        self.blue_slider.setMaximum(DEFAULT_COLOR_VALUES[2][1])
        self.blue_slider.setValue((DEFAULT_COLOR_VALUES[2][0], DEFAULT_COLOR_VALUES[2][1]))
        self.blue_slider.setTickPosition(QSlider.TicksBelow)
        self.blue_slider.setTickInterval(50)
        self.blue_slider.valueChanged.connect(self.blue_slider_moved)

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
        """
        update_color_button = QPushButton('Update', self)
        update_color_button.setToolTip('Update the color volume')
        update_color_button.clicked.connect(self.updateVolume)
        """
        ## Add rows ##
        subgrid_color.addRow('Red Slider', self.red_slider)
        subgrid_color.addRow('Green Slider', self.green_slider)
        subgrid_color.addRow('Blue Slider', self.blue_slider)

        subgrid_slider_buttons.addLayout(subgrid_color)
        subgrid_slider_buttons.addLayout(subgrid_buttons)
        """
        subgrid_color.addRow(QLabel('Set the color intensity ranges:'))
        subgrid_color.addRow('Red min', self.red_low)
        subgrid_color.addRow('Red max', self.red_high)
        
        subgrid_color.addRow('Green min', self.green_low)
        subgrid_color.addRow('Green max', self.green_high)
        subgrid_color.addRow('Blue min', self.blue_low)
        subgrid_color.addRow('Blue max', self.blue_high)
        subgrid_color.addRow(update_color_button)
        """
        self.subgrid_probes = QGridLayout()
        self.subgrid_probes_color = QVBoxLayout()
        
        for i, button in enumerate(self.probe_buttons):
            button.setToolTip('Annotate ' + button.text())
            button.clicked.connect(partial(self.selectProbe, button))
            self.subgrid_probes.addWidget(button, i//6, i % 6)
        
        self.subgrid_probes_color.addLayout(subgrid_slider_buttons)
        self.subgrid_probes_color.addStretch()
        self.subgrid_probes_color.addLayout(self.subgrid_probes)
        self.right_grid.addLayout(self.subgrid_probes_color)
        #grid.addLayout(subgrid,2,0)
        self.left_grid.addLayout(subgrid)
        #grid.addLayout(self.subgrid_probes_color, 0, 1)
        #grid.addLayout(subgrid_probes, 0, 1)

        self.main_grid.addLayout(self.left_grid)
        self.main_grid.addLayout(self.right_grid)
        self.current_directory = '/mnt/md0/data/opt/production'

        self.data_loaded = False

        self.selected_probe = None

        self.refresh_time=[]

        self.setLayout(self.main_grid)
        self.initial = True
        self.viewCoronal()
        self.show()
    
    # toggle the red check box, to filter/unfilter red
    def toggle_red(self):
        print('Working')
        if not self.is_red_checked:
            self.refreshImage(toggle='red')
            self.is_red_checked = True
            print('Checked')
        else:
            self.refreshImage(toggle='red_uncheck')
            self.is_red_checked = False
    
    # toggle the green check box, to filter/unfilter green
    def toggle_green(self):
        if not self.is_green_checked:
            self.refreshImage(toggle='green')
            self.is_green_checked = True
        else:
            self.refreshImage(toggle='green_uncheck')
            self.is_green_checked = False
    
    # toggle the blue check box, to filter/unfilter blue
    def toggle_blue(self):
        if not self.is_blue_checked:
            self.refreshImage(toggle='blue')
            self.is_blue_checked = True
        else:
            self.refreshImage(toggle='blue_uncheck')
            self.is_blue_checked = False
    
    # check if the red slider has been moved
    def red_slider_moved(self):
        if not self.is_red_checked:
            self.refreshImage(slider_moved='red', val=self.red_slider.value())
    
    # check if the green slider has been moved
    def green_slider_moved(self):
        if not self.is_green_checked:
            self.refreshImage(slider_moved='green', val=self.green_slider.value())
    
    # check if the blue slider has been moved
    def blue_slider_moved(self):
        if not self.is_blue_checked:
            self.refreshImage(slider_moved='blue', val=self.blue_slider.value())

    def keyPressEvent(self, e):

        if e.key() == Qt.Key_A:
            self.selectProbe(self.probe_buttons[0])
        if e.key() == Qt.Key_B:
            self.selectProbe(self.probe_buttons[1])
        if e.key() == Qt.Key_C:
            self.selectProbe(self.probe_buttons[2])
        if e.key() == Qt.Key_D:
            self.selectProbe(self.probe_buttons[3])
        if e.key() == Qt.Key_E:
            self.selectProbe(self.probe_buttons[4])
        if e.key() == Qt.Key_F:
            self.selectProbe(self.probe_buttons[5])
        if e.key() == Qt.Key_1:
            self.selectProbe(self.probe_buttons[6])
        if e.key() == Qt.Key_2:
            self.selectProbe(self.probe_buttons[7])
        if e.key() == Qt.Key_3:
            self.selectProbe(self.probe_buttons[8])
        if e.key() == Qt.Key_4:
            self.selectProbe(self.probe_buttons[9])
        if e.key() == Qt.Key_5:
            self.selectProbe(self.probe_buttons[10])
        if e.key() == Qt.Key_6:
            self.selectProbe(self.probe_buttons[11])
        if e.key() == Qt.Key_Backspace:
            self.deletePoint()

    def deletePoint(self):

        if self.selected_probe is not None:

            if self.current_view == 0:
                matching_index = self.annotations[(self.annotations.AP == self.slider.value()) &
                                                       (self.annotations.probe_name == self.selected_probe)].index.values
            elif self.current_view == 1:
                matching_index = self.annotations[(self.annotations.DV == self.slider.value()) &
                                                       (self.annotations.probe_name == self.selected_probe)].index.values
            elif self.current_view == 2:
                matching_index = self.annotations[(self.annotations.ML == self.slider.value()) &
                                                       (self.annotations.probe_name == self.selected_probe)].index.values

            if len(matching_index) > 0:
                self.annotations = self.annotations.drop(index=matching_index)

                self.saveData()

                self.refreshImage()

    def clickedOnImage(self , event):

        if self.data_loaded:
            x = int(event.pos().x()/SCALING_FACTOR)
            y = int(event.pos().y()/SCALING_FACTOR)

            # print('X: ' + str(x))
            # print('Y: ' + str(y))

            if self.selected_probe is not None:
                #print('updating volume')

                if self.current_view == 0:
                    AP = self.slider.value()
                    DV = y
                    ML = x
                    matching_index = self.annotations[(self.annotations.AP == AP) &
                                                       (self.annotations.probe_name ==
                                                        self.selected_probe)].index.values
                elif self.current_view == 1:
                    AP = y
                    DV = self.slider.value()
                    ML = x
                    matching_index = self.annotations[(self.annotations.DV == DV) &
                                                       (self.annotations.probe_name ==
                                                        self.selected_probe)].index.values
                elif self.current_view == 2:
                    AP = x
                    DV = y
                    ML = self.slider.value()
                    matching_index = self.annotations[(self.annotations.ML == ML) &
                                                       (self.annotations.probe_name ==
                                                        self.selected_probe)].index.values

                # Remove limitation of 1 point per probe per slice
                # if len(matching_index) > 0:
                #     self.annotations = self.annotations.drop(index=matching_index)

                self.annotations = self.annotations.append(pd.DataFrame(data = {'AP' : [AP],
                                    'ML' : [ML],
                                    'DV': [DV],
                                    'probe_name': [self.selected_probe]}),
                                    ignore_index=True)

                self.saveData()

                self.refreshImage()

    def selectProbe(self, b):

        for button in self.probe_buttons:
            button.setStyleSheet("background-color: white")

        b.setStyleSheet("background-color: " + self.color_map[b.text()])

        self.selected_probe = b.text()

    def sliderMoved(self):

        self.slider_values[self.current_view] = self.slider.value()
        self.refreshImage(change_view=True)
        #self.update_histogram()

    def viewCoronal(self):

        self.current_view = 0
        self.slider.setValue(self.slider_values[self.current_view])
        self.slider.setMaximum(528)
        self.width = int(456*SCALING_FACTOR)
        self.height = int(320*SCALING_FACTOR)
        #self.setGeometry(self.left, self.top, self.width, self.height)
        self.coronal_button.setStyleSheet("background-color: gray")
        self.horizontal_button.setStyleSheet("background-color: white")
        self.sagittal_button.setStyleSheet("background-color: white")

        if self.initial:
            self.refreshImage(change_view=False)
            self.initial = False
        else:
            self.refreshImage(change_view=True)

    def viewHorizontal(self):

        self.current_view = 1
        self.slider.setValue(self.slider_values[self.current_view])
        self.slider.setMaximum(320)
        #self.width = int(456*SCALING_FACTOR)
        #self.height = int(528*SCALING_FACTOR)
        #self.setGeometry(self.left, self.top, self.width, self.height)
        self.coronal_button.setStyleSheet("background-color: white")
        self.horizontal_button.setStyleSheet("background-color: gray")
        self.sagittal_button.setStyleSheet("background-color: white")
        
        if self.initial:
            self.refreshImage(change_view=False)
        else:
            self.refreshImage(change_view=True)

    def viewSagittal(self):

        self.current_view = 2
        self.slider.setValue(self.slider_values[self.current_view])
        self.slider.setMaximum(456)
        self.width = int(528*SCALING_FACTOR)
        self.height = int(320*SCALING_FACTOR)
        #self.setGeometry(self.left, self.top, self.width, self.height)
        self.coronal_button.setStyleSheet("background-color: white")
        self.horizontal_button.setStyleSheet("background-color: white")
        self.sagittal_button.setStyleSheet("background-color: gray")
        
        if self.initial:
            self.refreshImage(change_view=False)
        else:
            self.refreshImage(change_view=True)

    def create_histogram(self):
        inbins = np.arange(0, 40000, 10)

        for colori, carray in self.int_arrays.items():
            slicepixs, binins = np.histogram(carray[self.slider.value()], bins=inbins)
            allpixs, binins = np.histogram(carray, bins=inbins)
            
            self.sc.ax.semilogy(inbins[:-1], slicepixs+1, color=colori, linestyle='dashed', alpha=0.5)
            if colori == 'red':
                self.sc.ax.vlines([DEFAULT_COLOR_VALUES[0][0], DEFAULT_COLOR_VALUES[0][1]], color='red', ymin=0, ymax=10e8)
            elif colori == 'blue':
                self.sc.ax.vlines([DEFAULT_COLOR_VALUES[2][0], DEFAULT_COLOR_VALUES[2][1]], color='blue', ymin=0, ymax=10e8)
            #sc.ax.semilogy(inbins[:-1], allpixs+1, color=colori)
            self.sc.ax.set_xlim([0, 10000])
            self.sc.ax.set_xlabel('Intensity values')
            self.sc.ax.set_ylabel('Number of pixels')
            
            #graph.addPlot(x=inbins[:-1], y=slicepixs+1)
            #graph.addPlot(x=inbins[:-1], y=allpixs+1)

        self.subgrid_probes_color.insertWidget(1, self.sc)
        self.subgrid_probes_color.addStretch(3)

    def update_histogram(self):
        self.sc.ax.clear()
        self.create_histogram()
    
    # function that updates the image plane where certain events are triggered such as moving a slider or clicking a button
    def refreshImage(self, toggle='None', slider_moved='None', val=0, change_view=False):
        colors = ('darkred', 'orangered', 'goldenrod',
            'darkgreen', 'darkblue', 'blueviolet',
            'red','orange','yellow','green','blue','violet')
        """
        if reset:
            self.red_slider.setValue(DEFAULT_COLOR_VALUES[0][1])
            self.red_button.setChecked(False)
            self.red_checked = False

            self.green_slider.setValue(DEFAULT_COLOR_VALUES[1][1])
            self.green_button.setChecked(False)
            self.green_checked = False

            self.blue_slider.setValue(DEFAULT_COLOR_VALUES[2][1])
            self.blue_button.setChecked(False)
            self.blue_checked = False
        """
        if self.data_loaded:
            if self.current_view == 0:
                plane = self.volume[self.slider.value(),:,:,:]
            elif self.current_view == 1:
                plane = self.volume[:,self.slider.value(),:,:]
            elif self.current_view == 2:
                plane = self.volume[:,:,self.slider.value(),:]
                plane = np.swapaxes(plane, 0, 1) # plane.T
            # im8 = Image.fromarray(plane)
        else:
            # im8 = Image.fromarray(np.ones((self.height,self.width,3),dtype='uint8')*255)
            plane = np.ones((self.height,self.width,3),dtype='uint8')*255

        if change_view and not self.initial:
            # get existing values from sliders to update new view, change of view
            red_value = self.red_slider.value()
            red_clip = np.clip(self.int_arrays['red'], a_min=red_value[0], a_max=red_value[1]) - red_value[0]
            red_8 = (red_clip * 255. / (red_value[1] - red_value[0])).astype('uint8')

            green_value = self.green_slider.value()
            green_clip = np.clip(self.int_arrays['green'], a_min=green_value[0], a_max=green_value[1]) - green_value[0]
            green_8 = (green_clip * 255. / (green_value[1] - green_value[0])).astype('uint8')

            blue_value = self.blue_slider.value()
            blue_clip = np.clip(self.int_arrays['blue'], a_min=blue_value[0], a_max=blue_value[1]) - blue_value[0]
            blue_8 = (blue_clip * 255. / (blue_value[1] - blue_value[0])).astype('uint8')
            
            if self.current_view == 0:
                plane[:, :, 0] = red_8[self.slider.value(), :, :].copy()
                plane[:, :, 1] = green_8[self.slider.value(), :, :].copy()
                plane[:, :, 2] = blue_8[self.slider.value(), :, :].copy()
            elif self.current_view == 1:
                plane[:, :, 0] = red_8[:, self.slider.value(), :].copy()
                plane[:, :, 1] = green_8[:, self.slider.value(), :].copy()
                plane[:, :, 2] = blue_8[:, self.slider.value(), :].copy()
            elif self.current_view == 2:
                plane[:, :, 0] = red_8[:, :, self.slider.value()].transpose().copy()
                plane[:, :, 1] = green_8[:, :, self.slider.value()].transpose().copy()
                plane[:, :, 2] = blue_8[:, :, self.slider.value()].transpose().copy()

            if self.is_red_checked:
               self.red_old = plane[:, :, 0].copy()
               plane[:, :, 0] = 0

            if self.is_green_checked:
                self.green_old = plane[:, :, 1].copy()
                plane[:, :, 1] = 0

            if self.is_blue_checked:
                self.blue_old = plane[:, :, 2].copy()
                plane[:, :, 2] = 0

        if slider_moved == 'red': # if the red slider has moved, update plane 
            red_clip = np.clip(self.int_arrays['red'], a_min=val[0], a_max=val[1]) - val[0]
            red_8 = (red_clip * 255. / (val[1] - val[0])).astype('uint8')
            
            if self.current_view == 0:
                plane[:, :, 0] = red_8[self.slider.value(), :, :].copy()
            elif self.current_view == 1:
                plane[:, :, 0] = red_8[:, self.slider.value(), :].copy()
            elif self.current_view == 2:
                plane[:, :, 0] = red_8[:, :, self.slider.value()].transpose().copy()

        elif slider_moved == 'green': # same as above but with green slider
            green_clip = np.clip(self.int_arrays['green'], a_min=val[0], a_max=val[1]) - val[0]
            green_8 = (green_clip * 255. / (val[1] - val[0])).astype('uint8')

            if self.current_view == 0:
                plane[:, :, 1] = green_8[self.slider.value(), :, :].copy()
            elif self.current_view == 1:
                plane[:, :, 1] = green_8[:, self.slider.value(), :].copy()
            elif self.current_view == 2:
                plane[:, :, 1] = green_8[:, :, self.slider.value()].transpose().copy()
        elif slider_moved == 'blue': # same as above but with blue slider
            blue_clip = np.clip(self.int_arrays['blue'], a_min=val[0], a_max=val[1]) - val[0]
            blue_8 = (blue_clip * 255. / (val[1] - val[0])).astype('uint8')
            
            if self.current_view == 0:
                plane[:, :, 2] = blue_8[self.slider.value(), :, :].copy()
            elif self.current_view == 1:
                plane[:, :, 2] = blue_8[:, self.slider.value(), :].copy()
            elif self.current_view == 2:
                plane[:, :, 2] = blue_8[:, :, self.slider.value()].transpose().copy()
        
        if toggle == 'red': # red is checked, remove from plane, save previous for when red is unchecked
            self.red_old = plane[:, :, 0].copy()
            plane[:, :, 0] = 0
        elif toggle == 'green': # green is checked, same functionality as red
            self.green_old = plane[:, :, 1].copy()
            plane[:, :, 1] = 0
        elif toggle == 'blue': # blue is checked, same functionality as above
            self.blue_old = plane[:, :, 2].copy()
            plane[:, :, 2] = 0
        elif toggle == 'red_uncheck': # red is unchecked now, update plane with previous before red was checked
            print('Unchecked')
            plane[:, :, 0] = self.red_old.copy()
        elif toggle == 'green_uncheck': # green is unchecked now, update plane with previous before green was checked
            print('Unchecked')
            plane[:, :, 1] = self.green_old.copy()
        elif toggle == 'blue_uncheck': # blue is unchecked now, update plane with previous before blue was checked
            print('Unchecked')
            plane[:, :, 2] = self.blue_old.copy()
        
        image = plane.copy()
        height, width, channels = image.shape
        bytesPerLine = channels * width
        imQt = QImage(
            image.data, width, height, bytesPerLine, QImage.Format_RGB888
        )
        # imQt = QImage(ImageQt.ImageQt(im8))
        # imQt = imQt.convertToFormat(QImage.Format_RGB16)

        if self.data_loaded:
            for idx, row in self.annotations.iterrows():

                if self.current_view == 0:
                    shouldDraw = row.AP == self.slider.value()
                    # x = int(row.ML*SCALING_FACTOR)
                    # y = int(row.DV*SCALING_FACTOR)
                    x = row.ML
                    y = row.DV
                elif self.current_view == 1:
                    shouldDraw = row.DV == self.slider.value()
                    # x = int(row.ML*SCALING_FACTOR)
                    # y = int(row.AP*SCALING_FACTOR)
                    x = row.ML
                    y = row.AP
                elif self.current_view == 2:
                    shouldDraw = row.ML == self.slider.value()
                    # x = int(row.AP*SCALING_FACTOR)
                    # y = int(row.DV*SCALING_FACTOR)
                    x = row.AP
                    y = row.DV

                if shouldDraw:
                    color = QColor(self.color_map[row.probe_name])
                    point_size = int(2)

                    for j in range(x-point_size,x+point_size):
                        for k in range(y-point_size,y+point_size):
                            if pow(j-x,2) + pow(k-y,2) < 10:
                                imQt.setPixelColor(j,k,color)

        pxmap = QPixmap.fromImage(imQt).scaledToWidth(self.width).scaledToHeight(self.height)
        self.image.setPixmap(pxmap)
        self.setGeometry(self.left, self.top, self.width, self.height)

    def loadData(self):

        fname, filt = QFileDialog.getOpenFileName(self,
            caption='Select volume file',
            directory=self.current_directory)
            #filter='*.npy') # filter='*.mhd')

        print(fname)

        self.current_directory = os.path.dirname(fname)
        self.output_file = os.path.join(self.current_directory, 'probe_annotations.csv')

        if fname.split('.')[-1] == 'mhd':
            self.volume_type='TC'
            self.volume = self.loadVolume(fname)
            self.data_loaded = True
            self.setWindowTitle(os.path.basename(fname))
            if os.path.exists(self.output_file):
                self.annotations = pd.read_csv(self.output_file, index_col=0)
            else:
                self.annotations = pd.DataFrame(columns = ['AP','ML','DV', 'probe_name'])
            self.refreshImage()

        elif fname.split('.')[-1] == 'npy':
            self.volume_type='TC'
            self.volume = self.loadcolorVolume(fname)
            self.data_loaded = True
            self.setWindowTitle(os.path.basename(fname))
            if os.path.exists(self.output_file):
                self.annotations = pd.read_csv(self.output_file, index_col=0)
            else:
                self.annotations = pd.DataFrame(columns = ['AP','ML','DV', 'probe_name'])
            self.refreshImage()
        elif fname.split('.')[-1] == 'zip':
            self.volume_type = 'TC'
            self.loadVolumeFromZip(fname)
            self.data_loaded = True
            self.setWindowTitle(os.path.basename(fname))
            if os.path.exists(self.output_file):
                self.annotations = pd.read_csv(self.output_file, index_col=0)
            else:
                self.annotations = pd.DataFrame(columns = ['AP','ML','DV', 'probe_name'])
            self.refreshImage()
            self.create_histogram()

        else:
            print('invalid file')

    def loadVolumeFromZip(self, fname):
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

    def getColorVolume(self, rgb_levels=DEFAULT_COLOR_VALUES):
        level_adjusted_arrays = []
        for colori, int_level in zip(['red', 'green', 'blue'], rgb_levels):
            colarray = np.clip(self.int_arrays[colori], a_min=int_level[0], a_max=int_level[1]) - int_level[0]
            colarray = (colarray * 255. / (int_level[1] - int_level[0])).astype('uint8')
            level_adjusted_arrays.append(colarray)
        return np.stack(level_adjusted_arrays, axis=-1)

    def saveData(self):

        if self.data_loaded:
            self.annotations.to_csv(self.output_file)
    
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
    def loadVolume(self, fname, _dtype='u1'):

        resampled_image = sitk.ReadImage(fname)
        volume_temp = np.double(sitk.GetArrayFromImage(resampled_image)).transpose(2,1,0)

        upper_q=np.percentile(volume_temp,99)
        lower_q=np.percentile(volume_temp,50)

        #scaled & convert to uint8:
        volume_scaled = (volume_temp-lower_q)/(upper_q-lower_q)
        volume_scaled[volume_scaled<0]=0
        volume_scaled[volume_scaled>1]=1

        volume_sc_int8 = np.round(volume_scaled*255)

        dtype = np.dtype(_dtype)

        volume = np.asarray(volume_sc_int8, dtype)

        print("Data loaded.")

        return volume

    def loadcolorVolume(self, fname, _dtype='u1'): # LC added
        volume_temp = np.load(fname)
        # dtype = np.dtype(_dtype)
        # volume = np.asarray(volume_temp, dtype)
        print("Data loaded.")
        return volume_temp

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, self.ax = plt.subplots()
        super(MplCanvas, self).__init__(fig)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

        self.slider.setMinimum(0)
        self.slider.setMaximum(528)
        self.slider.setValue(DEFAULT_SLICE)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(50)
        self.slider.valueChanged.connect(self.sliderMoved)
        grid.addWidget(self.slider, 1, 0)

        self.slider_values = [DEFAULT_SLICE, DEFAULT_SLICE, DEFAULT_SLICE]

        subgrid = QGridLayout()

        self.probes = ('A1', 'B1', 'C1', 'D1', 'E1', 'F1',
                       'A2', 'B2', 'C2', 'D2', 'E2', 'F2')

        self.probe_map = {'Probe A1': 0, 'Probe B1': 1, 'Probe C1': 2, 'Probe D1' : 3,
                          'Probe E1': 4, 'Probe F1': 5,
                          'Probe A2': 6, 'Probe B2': 7, 'Probe C2': 8, 'Probe D2' : 9,
                          'Probe E2': 10, 'Probe F2': 11,
                     }

        self.color_map = {'Probe A1': 'darkred', 'Probe B1': 'cadetblue',
                     'Probe C1': 'goldenrod', 'Probe D1' : 'darkgreen',
                     'Probe E1': 'darkblue', 'Probe F1': 'blueviolet',
                     'Probe A2': 'red', 'Probe B2': 'darkturquoise',
                     'Probe C2': 'yellow', 'Probe D2' : 'green',
                     'Probe E2': 'blue', 'Probe F2': 'violet'}

        self.probe_buttons = [QPushButton('Probe ' + i) for i in self.probes]

        for i, button in enumerate(self.probe_buttons):
            button.setToolTip('Annotate ' + button.text())
            button.clicked.connect(partial(self.selectProbe, button))
            subgrid.addWidget(button, i//6, i % 6)

        self.coronal_button = QPushButton('Coronal', self)
        self.coronal_button.setToolTip('Switch to coronal view')
        self.coronal_button.clicked.connect(self.viewCoronal)

        self.horizontal_button = QPushButton('Horizontal', self)
        self.horizontal_button.setToolTip('Switch to horizontal view')
        self.horizontal_button.clicked.connect(self.viewHorizontal)

        self.sagittal_button = QPushButton('Sagittal', self)
        self.sagittal_button.setToolTip('Switch to sagittal view')
        self.sagittal_button.clicked.connect(self.viewSagittal)

        self.current_view = DEFAULT_VIEW

        subgrid.addWidget(self.coronal_button,2,0,1,2)
        subgrid.addWidget(self.horizontal_button,2,2,1,2)
        subgrid.addWidget(self.sagittal_button,2,4,1,2)

        save_button = QPushButton('Save', self)
        save_button.setToolTip('Save values as CSV')
        save_button.clicked.connect(self.saveData)

        load_button = QPushButton('Load', self)
        load_button.setToolTip('Load volume data')
        load_button.clicked.connect(self.loadData)

        subgrid.addWidget(save_button,3,4)
        subgrid.addWidget(load_button,3,5)

        grid.addLayout(subgrid,2,0)

        self.current_directory = '/mnt/md0/data/opt/production'

        self.data_loaded = False

        self.selected_probe = None

        self.refresh_time=[]

        self.setLayout(grid)
        self.viewCoronal()
        self.show()

    def keyPressEvent(self, e):

        if e.key() == Qt.Key_A:
            self.selectProbe(self.probe_buttons[0])
        if e.key() == Qt.Key_B:
            self.selectProbe(self.probe_buttons[1])
        if e.key() == Qt.Key_C:
            self.selectProbe(self.probe_buttons[2])
        if e.key() == Qt.Key_D:
            self.selectProbe(self.probe_buttons[3])
        if e.key() == Qt.Key_E:
            self.selectProbe(self.probe_buttons[4])
        if e.key() == Qt.Key_F:
            self.selectProbe(self.probe_buttons[5])
        if e.key() == Qt.Key_1:
            self.selectProbe(self.probe_buttons[6])
        if e.key() == Qt.Key_2:
            self.selectProbe(self.probe_buttons[7])
        if e.key() == Qt.Key_3:
            self.selectProbe(self.probe_buttons[8])
        if e.key() == Qt.Key_4:
            self.selectProbe(self.probe_buttons[9])
        if e.key() == Qt.Key_5:
            self.selectProbe(self.probe_buttons[10])
        if e.key() == Qt.Key_6:
            self.selectProbe(self.probe_buttons[11])
        if e.key() == Qt.Key_Backspace:
            self.deletePoint()

    def deletePoint(self):

        if self.selected_probe is not None:

            if self.current_view == 0:
                matching_index = self.annotations[(self.annotations.AP == self.slider.value()) &
                                                       (self.annotations.probe_name == self.selected_probe)].index.values
            elif self.current_view == 1:
                matching_index = self.annotations[(self.annotations.DV == self.slider.value()) &
                                                       (self.annotations.probe_name == self.selected_probe)].index.values
            elif self.current_view == 2:
                matching_index = self.annotations[(self.annotations.ML == self.slider.value()) &
                                                       (self.annotations.probe_name == self.selected_probe)].index.values

            if len(matching_index) > 0:
                self.annotations = self.annotations.drop(index=matching_index)

                self.saveData()

                self.refreshImage()

    def clickedOnImage(self , event):

        if self.data_loaded:
            x = int(event.pos().x()/SCALING_FACTOR)
            y = int(event.pos().y()/SCALING_FACTOR)

            # print('X: ' + str(x))
            # print('Y: ' + str(y))

            if self.selected_probe is not None:
                #print('updating volume')

                if self.current_view == 0:
                    AP = self.slider.value()
                    DV = y
                    ML = x
                    matching_index = self.annotations[(self.annotations.AP == AP) &
                                                       (self.annotations.probe_name ==
                                                        self.selected_probe)].index.values
                elif self.current_view == 1:
                    AP = y
                    DV = self.slider.value()
                    ML = x
                    matching_index = self.annotations[(self.annotations.DV == DV) &
                                                       (self.annotations.probe_name ==
                                                        self.selected_probe)].index.values
                elif self.current_view == 2:
                    AP = x
                    DV = y
                    ML = self.slider.value()
                    matching_index = self.annotations[(self.annotations.ML == ML) &
                                                       (self.annotations.probe_name ==
                                                        self.selected_probe)].index.values

                # Remove limitation of 1 point per probe per slice
                # if len(matching_index) > 0:
                #     self.annotations = self.annotations.drop(index=matching_index)

                self.annotations = self.annotations.append(pd.DataFrame(data = {'AP' : [AP],
                                    'ML' : [ML],
                                    'DV': [DV],
                                    'probe_name': [self.selected_probe]}),
                                    ignore_index=True)

                self.saveData()

                self.refreshImage()

    def selectProbe(self, b):

        for button in self.probe_buttons:
            button.setStyleSheet("background-color: white")

        b.setStyleSheet("background-color: " + self.color_map[b.text()])

        self.selected_probe = b.text()

    def sliderMoved(self):

        self.slider_values[self.current_view] = self.slider.value()
        self.refreshImage()

    def viewCoronal(self):

        self.current_view = 0
        self.slider.setValue(self.slider_values[self.current_view])
        self.slider.setMaximum(528)
        self.width = int(456*SCALING_FACTOR)
        self.height = int(320*SCALING_FACTOR)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.coronal_button.setStyleSheet("background-color: gray")
        self.horizontal_button.setStyleSheet("background-color: white")
        self.sagittal_button.setStyleSheet("background-color: white")
        self.refreshImage()

    def viewHorizontal(self):

        self.current_view = 1
        self.slider.setValue(self.slider_values[self.current_view])
        self.slider.setMaximum(320)
        self.width = int(456*SCALING_FACTOR)
        self.height = int(528*SCALING_FACTOR)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.coronal_button.setStyleSheet("background-color: white")
        self.horizontal_button.setStyleSheet("background-color: gray")
        self.sagittal_button.setStyleSheet("background-color: white")
        self.refreshImage()

    def viewSagittal(self):

        self.current_view = 2
        self.slider.setValue(self.slider_values[self.current_view])
        self.slider.setMaximum(456)
        self.width = int(528*SCALING_FACTOR)
        self.height = int(320*SCALING_FACTOR)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.coronal_button.setStyleSheet("background-color: white")
        self.horizontal_button.setStyleSheet("background-color: white")
        self.sagittal_button.setStyleSheet("background-color: gray")
        self.refreshImage()

    def refreshImage(self):
        colors = ('darkred', 'orangered', 'goldenrod',
            'darkgreen', 'darkblue', 'blueviolet',
            'red','orange','yellow','green','blue','violet')

        if self.data_loaded:
            if self.current_view == 0:
                plane = self.volume[self.slider.value(),:,:,:]
            elif self.current_view == 1:
                plane = self.volume[:,self.slider.value(),:,:]
            elif self.current_view == 2:
                plane = self.volume[:,:,self.slider.value(),:]
                plane = np.swapaxes(plane, 0, 1) # plane.T
            # im8 = Image.fromarray(plane)
        else:
            # im8 = Image.fromarray(np.ones((self.height,self.width,3),dtype='uint8')*255)
            plane = np.ones((self.height,self.width,3),dtype='uint8')*255

        image = plane.copy()
        height, width, channels = image.shape
        bytesPerLine = channels * width
        imQt = QImage(
            image.data, width, height, bytesPerLine, QImage.Format_RGB888
        )
        # imQt = QImage(ImageQt.ImageQt(im8))
        # imQt = imQt.convertToFormat(QImage.Format_RGB16)

        if self.data_loaded:
            for idx, row in self.annotations.iterrows():

                if self.current_view == 0:
                    shouldDraw = row.AP == self.slider.value()
                    # x = int(row.ML*SCALING_FACTOR)
                    # y = int(row.DV*SCALING_FACTOR)
                    x = row.ML
                    y = row.DV
                elif self.current_view == 1:
                    shouldDraw = row.DV == self.slider.value()
                    # x = int(row.ML*SCALING_FACTOR)
                    # y = int(row.AP*SCALING_FACTOR)
                    x = row.ML
                    y = row.AP
                elif self.current_view == 2:
                    shouldDraw = row.ML == self.slider.value()
                    # x = int(row.AP*SCALING_FACTOR)
                    # y = int(row.DV*SCALING_FACTOR)
                    x = row.AP
                    y = row.DV

                if shouldDraw:
                    color = QColor(self.color_map[row.probe_name])
                    point_size = int(2)

                    for j in range(x-point_size,x+point_size):
                        for k in range(y-point_size,y+point_size):
                            if pow(j-x,2) + pow(k-y,2) < 10:
                                imQt.setPixelColor(j,k,color)

        pxmap = QPixmap.fromImage(imQt).scaledToWidth(self.width).scaledToHeight(self.height)
        self.image.setPixmap(pxmap)
        self.setGeometry(self.left, self.top, self.width, self.height)

    def loadData(self):

        fname, filt = QFileDialog.getOpenFileName(self,
            caption='Select volume file',
            directory=self.current_directory,
            filter='*.npy') # filter='*.mhd')

        print(fname)

        self.current_directory = os.path.dirname(fname)
        self.output_file = os.path.join(self.current_directory, 'probe_annotations.csv')

        if fname.split('.')[-1] == 'mhd':
            self.volume_type='TC'
            self.volume = self.loadVolume(fname)
            self.data_loaded = True
            self.setWindowTitle(os.path.basename(fname))
            if os.path.exists(self.output_file):
                self.annotations = pd.read_csv(self.output_file, index_col=0)
            else:
                self.annotations = pd.DataFrame(columns = ['AP','ML','DV', 'probe_name'])
            self.refreshImage()

        elif fname.split('.')[-1] == 'npy':
            self.volume_type='TC'
            self.volume = self.loadcolorVolume(fname)
            self.data_loaded = True
            self.setWindowTitle(os.path.basename(fname))
            if os.path.exists(self.output_file):
                self.annotations = pd.read_csv(self.output_file, index_col=0)
            else:
                self.annotations = pd.DataFrame(columns = ['AP','ML','DV', 'probe_name'])
            self.refreshImage()

        else:
            print('invalid file')

    def saveData(self):

        if self.data_loaded:
            self.annotations.to_csv(self.output_file)

    def loadVolume(self, fname, _dtype='u1'):

        resampled_image = sitk.ReadImage(fname)
        volume_temp = np.double(sitk.GetArrayFromImage(resampled_image)).transpose(2,1,0)

        upper_q=np.percentile(volume_temp,99)
        lower_q=np.percentile(volume_temp,50)

        #scaled & convert to uint8:
        volume_scaled = (volume_temp-lower_q)/(upper_q-lower_q)
        volume_scaled[volume_scaled<0]=0
        volume_scaled[volume_scaled>1]=1

        volume_sc_int8 = np.round(volume_scaled*255)

        dtype = np.dtype(_dtype)

        volume = np.asarray(volume_sc_int8, dtype)

        print("Data loaded.")

        return volume

    def loadcolorVolume(self, fname, _dtype='u1'): # LC added
        volume_temp = np.load(fname)
        # dtype = np.dtype(_dtype)
        # volume = np.asarray(volume_temp, dtype)
        print("Data loaded.")
        return volume_temp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
