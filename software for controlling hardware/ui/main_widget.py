import os 

from dto.thread_sharing import (
     read_result, 
     read_start, 
     post_algorithm, 
     post_mode,
     post_leds,
     post_iso,
     post_shutter,
     post_result,
     post_currency,
     MODE_RECOGNITION,
     MODE_CAPTURING
)

from PySide2.QtCore import Slot, Qt, QTimer
from PySide2.QtGui import QFont, QPixmap, QImage
from PySide2.QtWidgets import (
     
    QLabel, 
    QWidget,
    QSizePolicy,
    QGridLayout,
    QComboBox,
    QSlider,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QCheckBox,
    QGroupBox,
    QRadioButton
)

class MainWidget(QWidget):

    DEFAULT_MESSAGE = 'No coins detected.\nNenalezeny mince.\n\nInsert coins.\nVložte mince.'

    CURRENCIES = {
        'Měna/Currency - CZK': 'CZK',
        'Měna/Currency - EUR': 'EUR'
    }

    ISO_MODES = {
        'ISO  10':  10,
        'ISO  50':  50,
        'ISO 100': 100,
        'ISO 200': 200,
        'ISO 300': 300,
        'ISO 400': 400,
        'ISO 500': 500,
        'ISO 800': 800,
    }

    SHUTTER_SPEEDS = {
        'Závěrka / Shutter: neurčeno' : None,
        'Závěrka / Shutter: 1 / 15' : 1 /  15,
        'Závěrka / Shutter: 1 / 30' : 1 /  30,
        'Závěrka / Shutter: 1 / 60' : 1 /  60,
        'Závěrka / Shutter: 1 / 120': 1 / 120,
        'Závěrka / Shutter: 1 / 200': 1 / 200,
        'Závěrka / Shutter: 1 / 250': 1 / 250,
        'Závěrka / Shutter: 1 / 400': 1 / 400,
        'Závěrka / Shutter: 1 / 500': 1 / 500,
    }

    LED_SETTINGS = {
        'Bez osvětlení / None': (False, False, False),
        'Vrchní / Top': (True, False, False),
        'Boční / Side': (False, True, False),
        'Spodní / Bottom': (False, False, True),
        'Vrchní + Boční / Top + Side': (True, True, False),
        'Vrchní + Spodní / Top + Bottom': (True, False, True),
        'Vše / All': (True, True, True),
    }
    
    def __init__(self, width = 1280, height = 720):
        QWidget.__init__(self)
        self.width = width
        self.height = height

        ##### Top level layout
        self.layout_vert = QVBoxLayout(self)
        self.setLayout(self.layout_vert)

        ##### Top bar
        self.top_bar = QWidget(self)

        self.top_bar_hbox = QHBoxLayout(self.top_bar)
        self.top_bar.setLayout(self.top_bar_hbox)

        self.top_bar_header = QLabel(self.top_bar)
        self.top_bar_header.setFont(QFont("Sans", 28, QFont.Medium) )
        self.top_bar_header.setAlignment(Qt.AlignLeft)
        self.top_bar_header.setText('Rozpoznávání mincí / Coin recognition')
        self.top_bar_header.resize(500, 80)
        self.top_bar_header.setMargin(5)
        
        self.top_bar_img_feec = QLabel(self.top_bar)
        map = QPixmap(os.path.dirname(os.path.realpath(__file__)) + '/../img/feec.png')
        map = map.scaled(500, 80, Qt.KeepAspectRatio)
        self.top_bar_img_feec.setAlignment(Qt.AlignRight)
        self.top_bar_img_feec.setPixmap(map)
        self.top_bar_img_feec.resize(500, 80)
        self.top_bar_img_feec.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.top_bar_img_cvg = QLabel(self.top_bar)
        map = QPixmap(os.path.dirname(os.path.realpath(__file__)) + '/../img/cvg.jpg')
        map = map.scaled(80, 80, Qt.KeepAspectRatio)
        self.top_bar_img_cvg.setPixmap(map)
        self.top_bar_img_cvg.resize(80, 80)
        self.top_bar_img_cvg.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.top_bar_hbox.addWidget(self.top_bar_header, Qt.AlignVCenter)
        self.top_bar_hbox.addWidget(self.top_bar_img_cvg)
        self.top_bar_hbox.addWidget(self.top_bar_img_feec)
        self.top_bar_hbox.setSpacing(20)

        self.layout_vert.addWidget(self.top_bar)

        ##### Tab widget
        self.tabs = QTabWidget(self)
        self.tabs.setFont(QFont("Sans", 20, QFont.Medium))
        self.tabs.currentChanged.connect(self.tab_changed)
        self.layout_vert.addWidget(self.tabs)

        ##### Demonstrator widget
        self.demonstrator = QWidget(self)
        self.demonstrator_grid = QGridLayout(self.demonstrator)
        self.demonstrator_grid.setHorizontalSpacing(25)
        self.demonstrator_grid.setVerticalSpacing(10)
        self.demonstrator_grid.setColumnStretch(0, 7)
        self.demonstrator_grid.setColumnStretch(1, 5)
        self.demonstrator_grid.setRowStretch(0,15)
        self.demonstrator_grid.setRowStretch(1,2)
        self.demonstrator_grid.setRowStretch(2,2)
        self.demonstrator_grid.setMargin(20)
        self.demonstrator.setLayout(self.demonstrator_grid)

        self.demonstrator_text = QLabel(self.demonstrator)
        self.demonstrator_text.setText(self.DEFAULT_MESSAGE)
        self.demonstrator_text.setFont(QFont("Monospace", 28, QFont.Medium) )
        self.demonstrator_text.setAlignment(Qt.AlignRight)
        self.demonstrator_grid.addWidget(self.demonstrator_text, 0, 1, Qt.AlignTop)

        self.demonstrator_image = QLabel(self.demonstrator)
        self.demonstrator_image.setStyleSheet("background-color:#dddddd;");
        self.demonstrator_grid.addWidget(self.demonstrator_image, 0, 0, 3, 1)

        self.demonstrator_controls = QWidget(self.demonstrator)
        self.demonstrator_controls_layout = QHBoxLayout(self.demonstrator_controls)
        self.demonstrator_controls.setLayout(self.demonstrator_controls_layout)
        self.demonstrator_grid.addWidget(self.demonstrator_controls, 2, 1)

        self.demonstrator_algorithm = QComboBox(self.demonstrator)
        self.demonstrator_algorithm.addItems(['Randomized test sequence', 'KDVN-algorithm', 'KDVN-algorithm-CNN (exp.)'])        
        self.demonstrator_algorithm.setCurrentText('KDVN-algorithm')
        self.demonstrator_algorithm.currentTextChanged.connect(self.algorithm_changed)
        self.demonstrator_controls_layout.addWidget(self.demonstrator_algorithm)
        self.demonstrator_controls_layout.setSpacing(50)
        self.demonstrator_controls_layout.setMargin(0)

        self.demonstrator_show_values = QCheckBox('Zobrazit výsledky na snímku\nShow results on the image', self.demonstrator)
        self.demonstrator_show_values.setStyleSheet("QCheckBox {font-size: 20px;} QCheckBox::indicator { width: 25px; height: 25px;font-size: 20px;}");
        self.demonstrator_show_values.setChecked(True)
        self.demonstrator_controls_layout.addWidget(self.demonstrator_show_values)

        self.demonstrator_currency = QComboBox(self.demonstrator)
        self.demonstrator_currency.addItems(list(self.CURRENCIES.keys()))
        self.demonstrator_currency.setCurrentText('CZK')
        self.demonstrator_currency.currentTextChanged.connect(self.currency_changed)
        self.demonstrator_grid.addWidget(self.demonstrator_currency, 1, 1)

        self.tabs.addTab(self.demonstrator, "Mince / Coins")

        ##### Image capture widget
        self.capture = QWidget(self)

        self.capture_grid = QGridLayout(self.capture)
        self.capture_grid.setHorizontalSpacing(25)
        self.capture_grid.setVerticalSpacing(25)
        self.capture_grid.setColumnStretch(0, 14)
        self.capture_grid.setColumnStretch(1, 5)
        self.capture_grid.setColumnStretch(2, 5)
        self.capture_grid.setRowStretch(0,1)
        self.capture_grid.setRowStretch(1,3)
        self.capture_grid.setRowStretch(2,2)
        self.capture_grid.setMargin(20)
        self.capture.setLayout(self.capture_grid)

        self.capture_text = QLabel(self.capture)
        self.capture_text.setText('Režim průběžného snímkování.\nContinuous capture mode.\n\nZvolte parametry kamery / Select camera parameters:')
        self.capture_text.setFont(QFont("Sans", 18, QFont.Medium) )
        self.capture_grid.addWidget(self.capture_text, 0, 1, 1, 2)

        self.capture_radios = QGroupBox(self.capture)
        self.capture_radios.setTitle('Osvětlení / Light settings')
        self.capture_grid.addWidget(self.capture_radios, 1, 1, 1, 2)

        self.capture_radios_layout = QVBoxLayout(self.capture_radios)
        self.capture_radios_layout.setSpacing(0)
        self.capture_radios_layout.setMargin(5)
        self.capture_radios.setLayout(self.capture_radios_layout)

        self.capture_radios_buttons = []
        for key in self.LED_SETTINGS:
            radio = QRadioButton(key, self.capture_radios)
            radio.toggled.connect(self.radio_toggled)
            self.capture_radios_layout.addWidget(radio)
            self.capture_radios_buttons.append(radio)
        self.capture_radios_buttons[-1].setChecked(True)

        self.capture_iso = QComboBox(self.capture)
        self.capture_iso.addItems(list(self.ISO_MODES.keys()))        
        self.capture_iso.setCurrentText('ISO 100')
        self.capture_iso.currentTextChanged.connect(self.iso_changed)
        self.capture_grid.addWidget(self.capture_iso, 2, 1, 1, 1)

        self.capture_shutter = QComboBox(self.capture)
        self.capture_shutter.addItems(list(self.SHUTTER_SPEEDS.keys()))
        self.capture_shutter.setCurrentText('Závěrka / Shutter: 1 / 500')
        self.capture_shutter.currentTextChanged.connect(self.shutter_changed)
        self.capture_grid.addWidget(self.capture_shutter, 2, 2, 1, 1)

        self.capture_image = QLabel(self.capture)
        self.capture_image.setStyleSheet("background-color:#dddddd;");
        self.capture_grid.addWidget(self.capture_image, 0, 0, 3, 1)

        self.tabs.addTab(self.capture, "Snímání / Capturing")

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_tick)
        self.timer.start(300)

        # Post the algorithm type immediately
        post_algorithm(self.demonstrator_algorithm.currentText())

    @Slot()
    def timer_tick(self):

        started = read_start()
        if started:
            mess = 'Hledám mince...\nLooking for coins...'
            self.demonstrator_text.setText(mess)

        data = read_result()
        if data is not None:

            if self.tabs.currentIndex() == MODE_RECOGNITION:

                # Update message
                if data['found']:
                    mess = "Nalezeno/found:\n"
                    for key in data['coins']:
                        if data['coins'][key] > 0:
                            mess += "{0}: {1}\n".format(key, data['coins'][key])
                else:
                    mess = self.DEFAULT_MESSAGE
                self.demonstrator_text.setText(mess)

                # Update image
                if self.demonstrator_show_values.isChecked():
                    self.demonstrator_image.setPixmap(self.__makeImage(data['image_annotated']))
                else:
                    self.demonstrator_image.setPixmap(self.__makeImage(data['image_original']))

            elif self.tabs.currentIndex() == MODE_CAPTURING:
                self.capture_image.setPixmap(self.__makeImage(data))


    @Slot()
    def algorithm_changed(self):
        post_algorithm(self.demonstrator_algorithm.currentText())

    @Slot()
    def currency_changed(self):
        post_currency(self.CURRENCIES[self.demonstrator_currency.currentText()])

    @Slot()
    def tab_changed(self, index):
        post_mode(index)
        post_result(None)

    @Slot()
    def iso_changed(self):
        post_iso(self.ISO_MODES[self.capture_iso.currentText()])

    @Slot()
    def radio_toggled(self):
        for radio in self.capture_radios_buttons:
            if radio.isChecked():
                post_leds(self.LED_SETTINGS[radio.text()])
                break

    @Slot()
    def shutter_changed(self):
        post_shutter(self.SHUTTER_SPEEDS[self.capture_shutter.currentText()])

    def __makeImage(self, npimg):

        sh = npimg.shape
        if len(sh) == 3:
            (height, width, _) = npimg.shape
            img = QImage(npimg.data, width, height, 3 * width, QImage.Format_RGB888)
            map = QPixmap()
            map.convertFromImage(img,Qt.ColorOnly);
        else:
            (height, width) = npimg.shape
            img = QImage(npimg.data, width, height, width, QImage.Format_Grayscale8)
            map = QPixmap()
            map.convertFromImage(img);
       

        return map.scaled(self.demonstrator_image.width(), self.demonstrator_image.height(), Qt.KeepAspectRatio)

