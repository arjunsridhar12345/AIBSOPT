"""

This app allows a user to browse through a Tissuecyte volume and mark probe track locations.

"""

import sys
from unittest.mock import DEFAULT

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget, QPushButton, QGridLayout, QCheckBox, QFormLayout, QFileDialog, QComboBox, QTableWidget, QTableWidgetItem
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtGui import QColor
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import numpy as np
from qtrangeslider import QRangeSlider
import os
from zipfile import ZipFile
import SimpleITK as sitk
from glob import glob
import pandas as pd
import statistics

# constants used by app
DEFAULT_SLICE = 200
DEFAULT_VIEW = 0
SCALING_FACTOR = 1.5
DEFAULT_COLOR_VALUES = [[0, 3000], [0, 3000], [0, 1000]]

class TissuecyteApp(QWidget):
    # initialize fields
    def __init__(self):
        super().__init__()
        self.title = 'Tissuecyte Annotation'
        self.left = 500
        self.top = 100
        # self.width = int(400*SCALING_FACTOR)
        # self.height = int(400*SCALING_FACTOR)
        self.width = int(456*SCALING_FACTOR) #default = coronal view
        self.height = int(320*SCALING_FACTOR)

        # initialize UI
        self.initUI()
    
    # initializes the UI for the app
    def initUI(self):
        self.setWindowTitle(self.title)
        # sets the rectangle for the UI app
        self.setGeometry(self.left, self.top, self.width, self.height)

        # 3 main layouts - left side, right side, and then main one
        self.leftLayout = QVBoxLayout()
        self.rightLayout = QVBoxLayout()
        self.bottomLayout = QHBoxLayout()
        self.mainLayout = QVBoxLayout()

        # create default image
        im8 = np.ones((self.height,self.width),dtype='uint8')*255
        self.image = pg.image(im8)
        self.image.ui.histogram.hide()
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()
        self.image.setObjectName('image')
        self.image.setImage(im8.transpose())

        self.pointsAdded = {'A1': [], 'A2': [], 'A3': [], 'A4': [], 'A5': [],
                            'B1': [], 'B2': [], 'B3': [], 'B4': [], 'B5': [],
                            'C1': [], 'C2': [], 'C3': [], 'C4': [], 'C5': [],
                            'D1': [], 'D2': [], 'D3': [], 'D4': [], 'D5': [],
                            'E1': [], 'E2': [], 'E3': [], 'E4': [], 'E5': [],
                            'F1': [], 'F2': [], 'F3': [], 'F4': [], 'F5': []}

        self.colors = {'A1': 'red', 'A2': 'dark red', 'A3': 'indian red', 'A4': 'orange red', 'A5': 'pale violet red',
                       'B1': 'blue', 'B2': 'dark blue', 'B3': 'deep sky blue', 'B4': 'dodger blue', 'B5': 'steel blue',
                       'C1': 'pink', 'C2': 'deep pink', 'C3': 'hot pink', 'C4': 'magenta', 'C5': 'dark magenta',
                       'D1': 'yellow', 'D2': 'light yellow', 'D3': 'gold', 'D4': 'goldenrod', 'D5': 'yellow green',
                       'E1': 'cyan', 'E2': 'dark cyan', 'E3': 'light cyan', 'E4': 'turquoise', 'E5': 'teal',
                       'F1': 'green', 'F2': 'dark green', 'F3': 'light green', 'F4': 'lawn green', 'F5': 'dark sea green'}

        self.rgb = {'A1': '(255, 0, 0)', 'A2': '(139, 0, 0)', 'A3': '(205, 92, 92)', 'A4': '(255, 69, 0)', 'A5': '(219, 112, 147)',
                    'B1': '(0, 0, 255)', 'B2': '(0, 0, 139)', 'B3': '( 0, 191, 255)', 'B4': '(30, 144, 255)', 'B5': '( 70, 130, 180)',
                    'C1': '(255, 192, 203)', 'C2': '(255, 20, 147)', 'C3': '(255, 105, 180)', 'C4': '(255, 0, 255)', 'C5': '(139, 0, 139)',
                    'D1': '(255, 255, 0)', 'D2': '(255, 255, 224)', 'D3': '(255, 215, 0)', 'D4': '(218, 165, 32)', 'D5': '(154, 205, 50)',
                    'E1': '( 0, 255, 255)', 'E2': '( 0, 139, 139)', 'E3': '(224, 255, 255)', 'E4': '(64, 224, 208)', 'E5': '(0, 128, 128)',
                    'F1': '( 0, 128, 0)', 'F2': '(0, 100, 0)', 'F3': '(144, 238, 144)', 'F4': '(124, 252, 0)', 'F5': '(143, 188, 143)'}

        self.mainLayout.addWidget(self.image)
        # add mouse click event
        self.image.getImageItem().mouseClickEvent = self.clickedOnImage

        # slider for slices
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(528)
        self.slider.setValue(DEFAULT_SLICE)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(50)
        self.slider.valueChanged.connect(self.sliderMoved)
        self.leftLayout.addWidget(self.slider)
        self.slider_values = [DEFAULT_SLICE, DEFAULT_SLICE, DEFAULT_SLICE]

        # probe and number drop down
        self.probeLayout = QGridLayout()

        self.probeNames = QComboBox()
        self.probeNames.addItem('Probe')
        self.probeNames.addItem('A')
        self.probeNames.addItem('B')
        self.probeNames.addItem('C')
        self.probeNames.addItem('D')
        self.probeNames.addItem('E')
        self.probeNames.addItem('F')
        self.probeNames.currentTextChanged.connect(self.probeChange)
        self.probeName = 'Probe'
        self.probeLayout.addWidget(self.probeNames, 0, 0)
        
        self.probeNumber = QComboBox()
        self.probeNumber.addItem('Trial')
        self.probeNumber.addItem('1')
        self.probeNumber.addItem('2')
        self.probeNumber.addItem('3')
        self.probeNumber.addItem('4')
        self.probeNumber.addItem('5')
        self.probeNumber.currentTextChanged.connect(self.trialChange)
        self.probeLayout.addWidget(self.probeNumber, 0, 1)
        self.trial = 'Trial'

        self.colorLabel = QLabel(text='Probe Color and Pattern')
        self.colorLabel.setStyleSheet('border: 1px solid black;')
        self.probeLayout.addWidget(self.colorLabel, 0, 2)
        self.leftLayout.addLayout(self.probeLayout)
        self.selectedProbe = self.probeName + self.trial

        # coronal button view
        self.coronalButton = QPushButton('Coronal', self)
        self.coronalButton.setToolTip('Switch to coronal view')
        self.coronalButton.clicked.connect(self.viewCoronal)

        # horizontal button view
        self.horizontalButton = QPushButton('Horizontal', self)
        self.horizontalButton.setToolTip('Switch to horizontal view')
        self.horizontalButton.clicked.connect(self.viewHorizontal)

        # sagittal button view
        self.sagittalButton = QPushButton('Sagittal', self)
        self.sagittalButton.setToolTip('Switch to sagittal view')
        self.sagittalButton.clicked.connect(self.viewSagittal)

        self.currentView = DEFAULT_VIEW

        # save button
        self.saveButton = QPushButton('Save', self)
        self.saveButton.setToolTip('Save values as CSV')
        self.saveButton.clicked.connect(self.saveData)

        # load button
        self.loadButton = QPushButton('Load Zip', self)
        self.loadButton.setToolTip('Load volume data')
        self.loadButton.clicked.connect(self.loadData)

        # delete button
        self.deleteButton = QPushButton('Delete Points', self)
        self.deleteButton.setToolTip('Delete Probe data')
        self.deleteButton.clicked.connect(self.deletePoint)
        self.deleteButton.setStyleSheet("color: red;font: bold 12px")

        # hide button
        self.hideButton = QPushButton('Hide Points', self)
        self.hideButton.setToolTip('Hide Points')
        self.hideButton.clicked.connect(self.hidePoints)

        # unhide button
        self.showButton = QPushButton('Show Points', self)
        self.showButton.setToolTip('Show Points')
        self.showButton.clicked.connect(self.showPoints)

        # point lock
        self.pointLockButton = QPushButton('Point Lock ON', self)
        self.pointLockButton.setToolTip('Toggle point lock')
        self.pointLockButton.clicked.connect(self.pointLockToggle)
        self.pointLockButton.setStyleSheet("background-color: rgb(170,0,0);color: white;font: bold 12px")

        # layout for above buttons
        self.buttonsLayout = QGridLayout()
        self.buttonsLayout.addWidget(self.coronalButton,2,0,1,2)
        self.buttonsLayout.addWidget(self.horizontalButton,2,2,1,2)
        self.buttonsLayout.addWidget(self.sagittalButton,2,4,1,2)
        self.buttonsLayout.addWidget(self.saveButton,3,2)
        self.buttonsLayout.addWidget(self.loadButton,3,3)
        self.buttonsLayout.addWidget(self.deleteButton, 3, 4)
        self.buttonsLayout.addWidget(self.hideButton, 3, 5)
        self.buttonsLayout.addWidget(self.showButton, 3, 6)
        self.buttonsLayout.addWidget(self.pointLockButton, 3, 7)

        # add buttons to left layout
        self.leftLayout.addLayout(self.buttonsLayout)

        # check boxes to toggle rgb values off/on
        self.redCheck = QCheckBox('Toggle Red')
        self.redOld = None
        self.isRedChecked = False
        self.redCheck.clicked.connect(self.toggleRed)

        self.greenCheck = QCheckBox('Toggle Green')
        self.greenOld = None
        self.isGreenChecked = False
        self.greenCheck.clicked.connect(self.toggleGreen)

        self.blueCheck = QCheckBox('Toggle Blue')
        self.blueOld = None
        self.isBlueChecked = False
        self.blueCheck.clicked.connect(self.toggleBlue)

        # clear button
        self.resetSlidersButton = QPushButton('Reset Sliders', self)
        self.resetSlidersButton.setToolTip('Reset Slider')
        self.resetSlidersButton.clicked.connect(self.resetSliders)

        # layout for check boxes
        self.checkLayout = QFormLayout()
        self.checkLayout.addWidget(self.redCheck)
        self.checkLayout.addWidget(self.greenCheck)
        self.checkLayout.addWidget(self.blueCheck)
        self.checkLayout.addWidget(self.resetSlidersButton)

        # rgb sliders to toggle value
        self.redSlider = QRangeSlider(Qt.Horizontal)
        self.redSlider.setMinimum(DEFAULT_COLOR_VALUES[0][0])
        self.redSlider.setMaximum(DEFAULT_COLOR_VALUES[0][1])
        self.redSlider.setValue((DEFAULT_COLOR_VALUES[0][0], DEFAULT_COLOR_VALUES[0][1]))
        self.redSlider.setTickPosition(QSlider.TicksBelow)
        self.redSlider.setTickInterval(50)
        self.redSlider.valueChanged.connect(self.redSliderMoved)

        self.greenSlider = QRangeSlider(Qt.Horizontal)
        self.greenSlider.setMinimum(DEFAULT_COLOR_VALUES[1][0])
        self.greenSlider.setMaximum(DEFAULT_COLOR_VALUES[1][1])
        self.greenSlider.setValue((DEFAULT_COLOR_VALUES[1][0], DEFAULT_COLOR_VALUES[1][1]))
        self.greenSlider.setTickPosition(QSlider.TicksBelow)
        self.greenSlider.setTickInterval(50)
        self.greenSlider.valueChanged.connect(self.greenSliderMoved)

        self.blueSlider = QRangeSlider(Qt.Horizontal)
        self.blueSlider.setMinimum(DEFAULT_COLOR_VALUES[2][0])
        self.blueSlider.setMaximum(DEFAULT_COLOR_VALUES[2][1])
        self.blueSlider.setValue((DEFAULT_COLOR_VALUES[2][0], DEFAULT_COLOR_VALUES[2][1]))
        self.blueSlider.setTickPosition(QSlider.TicksBelow)
        self.blueSlider.setTickInterval(50)
        self.blueSlider.valueChanged.connect(self.blueSliderMoved)

        # layout for sliders
        self.sliderLayout = QFormLayout()
        self.sliderLayout.addRow('Red Slider', self.redSlider)
        self.sliderLayout.addRow('Green Slider', self.greenSlider)
        self.sliderLayout.addRow('Blue Slider', self.blueSlider)

        # layout for sliders and checkboxes and probes
        self.checkSliderLayout = QHBoxLayout()
        self.checkSliderLayout.addLayout(self.sliderLayout)
        self.checkSliderLayout.addLayout(self.checkLayout)

        # add to right layout
        self.rightLayout.addLayout(self.checkSliderLayout)
        
        # add to bottom layout
        self.bottomLayout.addLayout(self.leftLayout)
        self.bottomLayout.addLayout(self.rightLayout)

        # add to main layout
        self.mainLayout.addLayout(self.bottomLayout)

        self.setLayout(self.mainLayout)
        self.showMaximized()
        # miscellanoues fieldss
        self.imgItems = []
        self.current_directory = '/mnt/md0/data/opt/production'

        self.data_loaded = False

        self.selected_probe = None

        self.point_lock = True

        self.refresh_time=[]
        self.initial = True
        self.viewCoronal()
        #self.show()
    
    # processes the probe name change from the drop down
    def probeChange(self, text):
        print(text)
        self.probeName = text
        self.updateProbeLabel()

    # processes the trial change from the drop down
    def trialChange(self, text):
        print(text)
        self.trial = text
        self.updateProbeLabel()
    
    def updateProbeHelper(self):
        #self.colorLabel.setText(self.colors[self.probeName + self.trial])
        self.colorLabel.setText('')
        self.colorLabel.setStyleSheet("background-color: rgb%s;color: black;font: 12px" %(self.rgb[self.probeName + self.trial]))

    def pointLockToggle(self):
        if self.point_lock:
            self.point_lock = False
            self.pointLockButton.setText('Point Lock OFF')
            self.pointLockButton.setStyleSheet("background-color: white;color: black;font: 12px")
        else:
            self.point_lock = True
            self.pointLockButton.setText('Point Lock ON')
            self.pointLockButton.setStyleSheet("background-color: rgb(170,0,0);color: white;font: bold 12px")

    def deletePoint(self):
        print('key pressed')
        if not self.point_lock:
            if self.probeName != 'Probe' and self.trial != 'Trial':
                self.selectedProbe = 'Probe' + ' ' + self.probeName +  self.trial
                if self.currentView == 0:
                    matching_index = self.annotations[(self.annotations.AP == self.slider.value()) &
                                                           (self.annotations.probe_name == self.selectedProbe)].index.values
                    print('yes delete')
                elif self.currentView == 1:
                    matching_index = self.annotations[(self.annotations.DV == self.slider.value()) &
                                                           (self.annotations.probe_name == self.selectedProbe)].index.values
                elif self.currentView == 2:
                    matching_index = self.annotations[(self.annotations.ML == self.slider.value()) &
                                                           (self.annotations.probe_name == self.selectedProbe)].index.values
    
                if len(matching_index) > 0:
                    self.annotations = self.annotations.drop(index=matching_index)
    
                    self.saveData()
                    
                    probe = self.selectedProbe[self.selectedProbe.index(' ') + 1:]
                    points = self.pointsAdded[probe]
                    view = self.image.getView()

                    for point in points:
                        view.removeItem(point)
                    
                    #self.refreshImage(value_draw=True)
    
    def updateProbeLabel(self):
        if self.probeName != 'Probe' and self.trial != 'Trial':
            self.updateProbeHelper()
    
    def resetSliders(self):
        self.redSlider.setValue((DEFAULT_COLOR_VALUES[0][0], DEFAULT_COLOR_VALUES[0][1]))
        self.greenSlider.setValue((DEFAULT_COLOR_VALUES[1][0], DEFAULT_COLOR_VALUES[1][1]))
        self.blueSlider.setValue((DEFAULT_COLOR_VALUES[2][0], DEFAULT_COLOR_VALUES[2][1]))
        
        self.refreshImage(change_view=True)

    # clears the annotations, does not remove them from csv
    def hidePoints(self):
        view = self.image.getView()

        while len(view.addedItems) > 3:
            view.removeItem(view.addedItems.pop())


    # shows the hidden points
    def showPoints(self):
        self.refreshImage(value_draw=True)
    
    def clickedOnImage(self , event):
        print('Click')
        event.accept()
        if not self.point_lock:
            if self.data_loaded:
                x = int(event.pos().x())
                y = int(event.pos().y())

                # print('X: ' + str(x))
                # print('Y: ' + str(y))

                if self.probeName != 'Probe' and self.trial != 'Trial':
                    self.selectedProbe = 'Probe' + ' ' + self.probeName +  self.trial
                    print(self.trial)
                    #print('updating volume')
                    if self.currentView == 0:
                        AP = self.slider.value()
                        DV = y
                        ML = x
                        matching_index = self.annotations[(self.annotations.AP == AP) &
                                                           (self.annotations.probe_name ==
                                                            self.selectedProbe)].index.values
                    elif self.currentView == 1:
                        AP = y
                        DV = self.slider.value()
                        ML = x
                        matching_index = self.annotations[(self.annotations.DV == DV) &
                                                           (self.annotations.probe_name ==
                                                            self.selectedProbe)].index.values
                    elif self.currentView == 2:
                        AP = x
                        DV = y
                        ML = self.slider.value()
                        matching_index = self.annotations[(self.annotations.ML == ML) &
                                                           (self.annotations.probe_name ==
                                                            self.selectedProbe)].index.values

                    # Remove limitation of 1 point per probe per slice
                    # if len(matching_index) > 0:
                    #     self.annotations = self.annotations.drop(index=matching_index)
                
                    self.annotations = self.annotations.append(pd.DataFrame(data = {'AP' : [AP],
                                        'ML' : [ML],
                                        'DV': [DV],
                                        'probe_name': [self.selectedProbe]}),
                                        ignore_index=True)

                    self.saveData()
                    self.refreshImage(click_draw=True, posx=x, posy=y)

    # updates image when slice slider is moved
    def sliderMoved(self):
        view = self.image.getView()
        while len(view.addedItems) > 3:
            view.removeItem(view.addedItems.pop())
        
        self.pointsAdded.update((key, []) for key in self.pointsAdded)

        self.slider_values[self.currentView] = self.slider.value()
        self.refreshImage(change_view=True, value_draw=True)

    # toggle the red check box, to filter/unfilter red
    def toggleRed(self):
        print('Working')
        if not self.isRedChecked:
            self.refreshImage(toggle='red')
            self.isRedChecked = True
            print('Checked')
        else:
            self.refreshImage(toggle='red_uncheck')
            self.isRedChecked = False
    
    # toggle the green check box, to filter/unfilter green
    def toggleGreen(self):
        if not self.isGreenChecked:
            self.refreshImage(toggle='green')
            self.isGreenChecked = True
        else:
            self.refreshImage(toggle='green_uncheck')
            self.isGreenChecked = False
    
    # toggle the blue check box, to filter/unfilter blue
    def toggleBlue(self):
        if not self.isBlueChecked:
            self.refreshImage(toggle='blue')
            self.isBlueChecked = True
        else:
            self.refreshImage(toggle='blue_uncheck')
            self.isBlueChecked = False
    
    # check if the red slider has been moved
    def redSliderMoved(self):
        if not self.isRedChecked:
            self.refreshImage(slider_moved='red', val=self.redSlider.value())
    
    # check if the green slider has been moved
    def greenSliderMoved(self):
        if not self.isGreenChecked:
            self.refreshImage(slider_moved='green', val=self.greenSlider.value())
    
    # check if the blue slider has been moved
    def blueSliderMoved(self):
        if not self.isBlueChecked:
            self.refreshImage(slider_moved='blue', val=self.blueSlider.value())

    # cornonal view
    def viewCoronal(self):
        self.currentView = 0
        self.slider.setValue(self.slider_values[self.currentView])
        self.slider.setMaximum(528)
        self.width = int(456*SCALING_FACTOR)
        self.height = int(320*SCALING_FACTOR)
        #self.setGeometry(self.left, self.top, self.width, self.height)
        self.coronalButton.setStyleSheet("background-color: gray")
        self.horizontalButton.setStyleSheet("background-color: white")
        self.sagittalButton.setStyleSheet("background-color: white")

        if self.initial: # initial view on loadup
            self.refreshImage(change_view=False)
            self.initial = False
        else:
            self.hidePoints()
            self.refreshImage(change_view=True, value_draw=True)
    
    # horizontal view
    def viewHorizontal(self):
        self.currentView = 1
        self.slider.setValue(self.slider_values[self.currentView])
        self.slider.setMaximum(320)
        #self.width = int(456*SCALING_FACTOR)
        #self.height = int(528*SCALING_FACTOR)
        #self.setGeometry(self.left, self.top, self.width, self.height)
        self.coronalButton.setStyleSheet("background-color: white")
        self.horizontalButton.setStyleSheet("background-color: gray")
        self.sagittalButton.setStyleSheet("background-color: white")
        
        if self.initial:
            self.refreshImage(change_view=False)
        else:
            self.hidePoints()
            self.refreshImage(change_view=True, value_draw=True)

    # sagittal view
    def viewSagittal(self):

        self.currentView = 2
        self.slider.setValue(self.slider_values[self.currentView])
        self.slider.setMaximum(456)
        self.width = int(528*SCALING_FACTOR)
        self.height = int(320*SCALING_FACTOR)
        #self.setGeometry(self.left, self.top, self.width, self.height)
        self.coronalButton.setStyleSheet("background-color: white")
        self.horizontalButton.setStyleSheet("background-color: white")
        self.sagittalButton.setStyleSheet("background-color: gray")
        
        if self.initial:
            self.refreshImage(change_view=False)
        else:
            self.hidePoints()
            self.refreshImage(change_view=True, value_draw=True)
    
    # helper function to threshold image
    def imageClip(self, color, val):
        if color == 'red':
            array = self.int_arrays['red']
        elif color == 'green':
            array = self.int_arrays['green']
        elif color == 'blue':
            array = self.int_arrays['blue']

        if self.currentView == 0:
            clip = np.clip(array[self.slider.value(), :, :], a_min=val[0], a_max=val[1]) - val[0]
        elif self.currentView == 1:
            clip = np.clip(array[:, self.slider.value(), :], a_min=val[0], a_max=val[1]) - val[0]
        elif self.currentView == 2:
            clip = np.clip(array[:, :, self.slider.value()].transpose(), a_min=val[0], a_max=val[1]) - val[0]

        clip_8 = (clip * 255. / (val[1] - val[0])).astype('uint8')
        
        return clip_8

    # helper function to draw points from click or dataframe
    def drawPointsHelper(self, j, k, color, probe):
        c = QtWidgets.QGraphicsRectItem(j, k, 0.05, 0.05)
        c.setPen(QtGui.QPen(color, 1))
        #c.setBrush(QtGui.QBrush(color, Qt.SolidPattern))
       
        view = self.image.getView()
        view.addItem(c)
        self.pointsAdded[probe].append(c)
    
    def drawPoints(self, color, probe_let, probe_trial, posx, posy, x, y):
        point_size = int(2)
        
        if posx is not None and posy is not None: # clicking has occurred, so use posx and posy, mouse coords
             draw = False
             for j in range(posx-point_size,posx+point_size):
                 for k in range(posy-point_size,posy+point_size):
                     if pow(j-x,2) + pow(k-y,2) < 10:
                         #print(j, k)
                         if not draw:
                            self.drawPointsHelper(j, k, color, probe_let + probe_trial)
                            #draw = True
        else: # no clicking has occurred, so use x and y from annotations dataframe - see refresh image function
            draw = False
            for j in range(x-point_size,x+point_size):
                 for k in range(y-point_size,y+point_size):
                     if pow(j-x,2) + pow(k-y,2) < 10:
                         #print(j, k)
                         if not draw:
                            self.drawPointsHelper(j, k, color, probe_let + probe_trial)
                            #draw = True

    # function that updates the image plane where certain events are triggered such as moving a slider or clicking a button
    # change view - save current settings of sliders, checks, etc. and then change to new view with these settings
    # value_draw - draw points from annotations csv
    # click_draw - draw point from click from mouse
    # posx - x postiion of mouse click
    # posy - y position of mouse click
    def refreshImage(self, toggle='None', slider_moved='None', val=0, change_view=False, value_draw=False, click_draw=False, posx=None, posy=None):
        plane = None

        if self.data_loaded:
            if self.currentView == 0:
                if self.slider.value() < self.slider.maximum():
                    print(self.volume.shape)
                    plane = self.volume[self.slider.value(),:,:]
            elif self.currentView == 1:
                if self.slider.value() < self.slider.maximum():
                    plane = self.volume[:, self.slider.value(),:,:]
            elif self.currentView == 2:
                if self.slider.value() < self.slider.maximum():
                    plane = self.volume[:, :, self.slider.value(), :]
                    plane = np.swapaxes(plane, 0, 1) # plane.T
            # im8 = Image.fromarray(plane)
        else:
            # im8 = Image.fromarray(np.ones((self.height,self.width,3),dtype='uint8')*255)
            plane = np.ones((self.height,self.width,3),dtype='uint8')*255
        
        if plane is not None:
            if change_view and not self.initial:
                # get existing values from sliders to update new view, change of view
                plane[:, :, 0] = self.imageClip('red', self.redSlider.value())
                plane[:, :, 1] = self.imageClip('green', self.greenSlider.value())
                plane[:, :, 2] = self.imageClip('blue', self.blueSlider.value())

                if self.isRedChecked:
                   self.redOld = plane[:, :, 0].copy()
                   plane[:, :, 0] = 0

                if self.isGreenChecked:
                    self.greenOld = plane[:, :, 1].copy()
                    plane[:, :, 1] = 0

                if self.isBlueChecked:
                    self.blueOld = plane[:, :, 2].copy()
                    plane[:, :, 2] = 0
        
            if slider_moved == 'red': # if the red slider has moved, update plane 
                plane[:, :, 0] = self.imageClip('red', val)
            elif slider_moved == 'green': # same as above but with green slider
                plane[:, :, 1] = self.imageClip('green', val)
            elif slider_moved == 'blue': # same as above but with blue slider
                plane[:, :, 2] = self.imageClip('blue', val)

            if toggle == 'red': # red is checked, remove from plane, save previous for when red is unchecked
                self.redOld = plane[:, :, 0].copy()
                plane[:, :, 0] = 0
            elif toggle == 'green': # green is checked, same functionality as red
                self.greenOld = plane[:, :, 1].copy()
                plane[:, :, 1] = 0
            elif toggle == 'blue': # blue is checked, same functionality as above
                self.blueOld = plane[:, :, 2].copy()
                plane[:, :, 2] = 0
            elif toggle == 'red_uncheck': # red is unchecked now, update plane with previous before red was checked
                print('Unchecked')
                plane[:, :, 0] = self.redOld.copy()
            elif toggle == 'green_uncheck': # green is unchecked now, update plane with previous before green was checked
                print('Unchecked')
                plane[:, :, 1] = self.greenOld.copy()
            elif toggle == 'blue_uncheck': # blue is unchecked now, update plane with previous before blue was checked
                print('Unchecked')
                plane[:, :, 2] = self.blueOld.copy()

            rotate = np.rot90(plane)
            flip = np.flipud(rotate)
            self.image.setImage(flip, levels=(0, 255))

            if self.data_loaded:
                for idx, row in self.annotations.iterrows():
                    if self.currentView == 0:
                        shouldDraw = row.AP == self.slider.value()
                        # x = int(row.ML*SCALING_FACTOR)
                        # y = int(row.DV*SCALING_FACTOR)
                        x = row.ML
                        y = row.DV
                    elif self.currentView == 1:
                        shouldDraw = row.DV == self.slider.value()
                        # x = int(row.ML*SCALING_FACTOR)
                        # y = int(row.AP*SCALING_FACTOR)
                        x = row.ML
                        y = row.AP
                    elif self.currentView == 2:
                        shouldDraw = row.ML == self.slider.value()
                        # x = int(row.AP*SCALING_FACTOR)
                        # y = int(row.DV*SCALING_FACTOR)
                        x = row.AP
                        y = row.DV

                    if shouldDraw:
                        probe = row.probe_name[row.probe_name.index(' ') + 1:]
                        probe_let = probe[0]
                        probe_num = probe[1]

                        if value_draw or click_draw:
                            color = QColor(self.colors[probe_let + probe_num])
                            self.drawPoints(color, probe_let, probe_num, posx, posy, x, y)

    # loads data from file selected
    def loadData(self):
        fname, filt = QFileDialog.getOpenFileName(self,
            caption='Select volume file',
            directory=self.current_directory)
            #filter='*.npy') # filter='*.mhd')

        print(fname)

        self.current_directory = os.path.dirname(fname)
        self.output_file = os.path.join(self.current_directory, 'probe_annotations.csv')

        if fname.split('.')[-1] == 'zip':
            self.volume_type = 'TC'
            self.loadVolumeFromZip(fname)
            self.data_loaded = True
            self.setWindowTitle(os.path.basename(fname))
            if os.path.exists(self.output_file):
                self.annotations = pd.read_csv(self.output_file, index_col=0)
            else:
                self.annotations = pd.DataFrame(columns = ['AP','ML','DV', 'probe_name'])
            self.refreshImage(value_draw=True)
            #self.create_histogram()

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
            print(intensity_arrays[imcolor].shape)
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
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = TissuecyteApp()
    w.show()
    sys.exit(app.exec_())