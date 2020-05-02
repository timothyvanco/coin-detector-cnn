import _thread
import time
import sys
import os

from demonstrator import demonstrator_proc
from PySide2.QtWidgets import QApplication
from ui.main_widget import MainWidget

# Algorithm and demonstrator processing
# Runs in a separate thread
_thread.start_new_thread(demonstrator_proc, ())

# Qt application bootstrap
w = 1280
h = 720
app = QApplication(sys.argv)
widget = MainWidget(w, h)
widget.resize(w, h)
widget.show()
sys.exit(app.exec_())

