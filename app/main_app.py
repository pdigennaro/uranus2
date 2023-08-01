import sys
import time
import torch
from PyQt6 import QtWidgets
from numpy import float64
from torch import tensor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import pandas as pd
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QTextEdit, QCheckBox
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
import pickle as pk
from pyproj import Proj
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from config import CASE_UNKNOWN, CASE_FIXED_WING, CASE_MAVIC_PRO, CASE_PHANTOM_4_PRO, CASE_MAVIC2, \
    CASE_PHANTOM4PRO_MAVIC2, CASE_PHANTOM4PRO_MAVICPRO, Net, COLS_TO_STANDARDIZE, MLP_INPUT_COLUMNS, MLP_CLASSIFIER_INPUT_COLUMNS

categories = [[CASE_UNKNOWN],[CASE_FIXED_WING],[CASE_MAVIC_PRO],[CASE_PHANTOM_4_PRO],[CASE_MAVIC2],[CASE_PHANTOM4PRO_MAVIC2],[CASE_PHANTOM4PRO_MAVICPRO]]
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(categories)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

WEIGHTS_PATH = "../results/classifier/uranus_e50_m5_PRE_CLASS_PD.pt"

RESULTS_BASE_PATH = "../results"
input_test_csv = "../datasets/merge/dataset_UTM.csv"
p = Proj(proj='utm',zone=31,ellps='WGS84', preserve_units=False)

model = Net(n_features=12).to(device) # define the network

model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval()

scaler = pk.load(open(RESULTS_BASE_PATH + '/scaler.pkl', 'rb'))

def classificationToString(classify):
    if classify == CASE_UNKNOWN:
        return "UKNOWN"
    if classify == CASE_FIXED_WING:
        return "FIXED WING"
    if classify == CASE_MAVIC_PRO:
        return "MAVIC PRO"
    if classify == CASE_PHANTOM_4_PRO:
        return "PHANTOM 4 PRO"
    if classify == CASE_MAVIC2:
        return "MAVIC 2"
    if classify == CASE_PHANTOM4PRO_MAVIC2:
        return "PHANTOM 4 PRO + MAVIC 2"
    if classify == CASE_PHANTOM4PRO_MAVICPRO:
        return "PHANTOM 4 PRO + MAVIC PRO"

def imageFromClassification(classify):
    if classify == CASE_UNKNOWN:
        return "imgs/unknown.jpg"
    if classify == CASE_FIXED_WING:
        return "imgs/rcs_parrot.png"
    if classify == CASE_MAVIC_PRO:
        return "imgs/rcs_mavicpro.png"
    if classify == CASE_PHANTOM_4_PRO:
        return "imgs/rcs_phantom.png"
    if classify == CASE_MAVIC2:
        return "imgs/rcs_mavic2.png"
    if classify == CASE_PHANTOM4PRO_MAVIC2:
        return "imgs/rcs_phantom_mavic2.png"
    if classify == CASE_PHANTOM4PRO_MAVICPRO:
        return "imgs/rcs_phantom_mavicpro.png"

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.form_widget = FormWidget(self)

        self.realTime = False
        self.second_miss = 0
        self.total_counter = 0

        _widget = QWidget()

        self.btn1 = QPushButton("Execute")
        self.btn1.clicked.connect(self.buttonAction)

        self.lblClassification = QLabel("UNKNOWN")
        self.lblClassification.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lblClassification.setStyleSheet("QLabel{font-size: 30pt; font-weight: bold;}")

        self.lblImage = QLabel(self)
        pixmap = QPixmap('imgs/unknown.jpg')
        self.lblImage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lblImage.setPixmap(pixmap)

        vboxResults = QVBoxLayout()
        vboxResults.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.buttonCheckBox = QCheckBox("Realtime show")
        self.buttonCheckBox.stateChanged.connect(lambda: self.btnstate(self.buttonCheckBox))

        self.combobox2 = QComboBox()
        self.combobox2.addItems(['Scenario_1_1', 'Scenario_1_2_b', 'Scenario_1_3', 'Scenario_1_4', 'Scenario_2_1', 'Scenario_2_2', 'Scenario_Parrot_a'])
        self.combobox2.currentTextChanged.connect(self.text_changed)
        self.combobox2.currentIndexChanged.connect(self.index_changed)

        self.textEdit = QTextEdit()

        vboxResults.addWidget(self.lblImage)
        vboxResults.addWidget(self.lblClassification)
        vboxResults.addWidget(self.combobox2)
        vboxResults.addWidget(self.textEdit)
        vboxResults.addWidget(self.buttonCheckBox, alignment=Qt.AlignmentFlag.AlignCenter)

        self.btn2 = QPushButton("Clear")
        self.btn2.clicked.connect(self.buttonAction2)

        hlayout = QHBoxLayout()
        hlayout.addWidget(self.form_widget, stretch=5)
        hlayout.addLayout(vboxResults, stretch=2)

        _layout = QVBoxLayout(_widget)
        _layout.addLayout(hlayout)
        _layout.addWidget(self.btn1)
        _layout.addWidget(self.btn2)

        self.setCentralWidget(_widget)

        self.full_set = pd.read_csv(input_test_csv)

        AlviraTracksTrack_ClassificationENC = LabelEncoder()
        self.full_set["AlviraTracksTrack_Classification"] = AlviraTracksTrack_ClassificationENC.fit_transform(self.full_set["AlviraTracksTrack_Classification"])

        ArcusTracksTrack_ClassificationENC = LabelEncoder()
        self.full_set["ArcusTracksTrack_Classification"] = ArcusTracksTrack_ClassificationENC.fit_transform(self.full_set["ArcusTracksTrack_Classification"])

        self.test_set = self.full_set.loc[self.full_set["scenario_name"] == "Scenario_1_1"].copy()

        self.regressor = pk.load(open(RESULTS_BASE_PATH + '/regressor/regressor.pkl', 'rb'))
        print(self.regressor)

    def btnstate(self, b):
        print(b.isChecked())
        self.realTime = b.isChecked()

    def text_changed(self, s):
        self.form_widget.browser.page().runJavaScript("clearPolys();")
        self.test_set = self.full_set.loc[self.full_set["scenario_name"] == s].copy()
        print(f"{s} loaded!")

    def index_changed(self, index):
        print("Index changed", index)
    def buttonAction2(self, click):
        print(f"click {click}")
        self.form_widget.browser.page().runJavaScript("clearPolys();")

        self.second_miss = 0
        self.total_counter = 0

    def appendToTextBox(self, texto):
        self.textEdit.append(texto)

    def buttonAction(self, click):
        print(f"click {click}")

        if self.realTime:
            self.form_widget.browser.page().runJavaScript("addLayerGroup();")

        #for ii in range(len(self.test_set)):
        self.total_counter += 1
        ii = self.total_counter
        print(self.total_counter)

        if self.realTime:
            self.form_widget.browser.page().runJavaScript("clearPoints();")

        if ii % 1 == 0:
            print(f"--- TEST {ii} ---")
            self.appendToTextBox(f"--- TEST {ii} ---")

        curr_dataframe = self.test_set.iloc[ii, :].copy()
        scaled_row = self.test_set.iloc[[ii]].copy()
        scaled_row[COLS_TO_STANDARDIZE] = scaler.transform(scaled_row[COLS_TO_STANDARDIZE])

        inputs = scaled_row[MLP_CLASSIFIER_INPUT_COLUMNS].values
        inputs = torch.from_numpy(inputs.astype(float64))
        inputs = inputs.to(device)
        inputs = inputs.float()
        inputs = inputs.reshape(inputs.shape[0], -1)

        y_pred = model(inputs)
        #print(f"ypred: {y_pred}")
        y_pred = torch.argmax(y_pred, 1)
        print(f"value: {y_pred}")
        class_true_val = curr_dataframe["reference_classification"]
        print(f"true val: {class_true_val}")

        textclass = classificationToString(y_pred)
        self.lblClassification.setText(textclass)

        imgPath = imageFromClassification(y_pred)
        pixmap = QPixmap(imgPath)
        self.lblImage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lblImage.setPixmap(pixmap)

        ground_truth = curr_dataframe[['utm_x', 'utm_y', 'utm_x2', 'utm_y2']].values

        (real_x, real_y) = p(ground_truth[0], ground_truth[1], inverse=True)
        (real_x2, real_y2) = p(ground_truth[2], ground_truth[3], inverse=True)

        if self.realTime:
            print(f"addRealDronesPoints({real_y}, {real_x}, {real_y2}, {real_x2});")
            self.form_widget.browser.page().runJavaScript(f"addRealDronesPoints({real_y}, {real_x}, {real_y2}, {real_x2});")

        identified = True

        if curr_dataframe["alvira_utm_x"] == 0 and curr_dataframe["arcus_utm_x"] == 0:
            print("not enough infos!")
            identified = False

        if y_pred == 0:
            print("No detection!")
            identified = False

        if identified:
            if y_pred[0] > 0:
                drones = 1 if y_pred[0] < 5 else 2
            else:
                drones = 0

            print(f"drones: {drones}")

            curr_dataframe[['drones']] = [drones]
            #print(f"curr_dataframe:\n{curr_dataframe}")
            print(self.regressor)

            pred = self.regressor.predict([curr_dataframe[['AlviraTracksTrackPosition_Altitude',
                                                            'AlviraTracksTrackVelocity_Azimuth',
                                                            'AlviraTracksTrackVelocity_Elevation',
                                                            'AlviraTracksTrackVelocity_Speed',
                                                            'AlviraTracksTrack_Classification',
                                                            'AlviraTracksTrack_Score',
                                                            'ArcusTracksTrackPosition_Altitude',
                                                            'ArcusTracksTrackVelocity_Azimuth',
                                                            'ArcusTracksTrackVelocity_Elevation',
                                                            'ArcusTracksTrackVelocity_Speed',
                                                            'ArcusTracksTrack_Classification',
                                                            'ArcusTracksTrack_Score',
                                                            'drones',
                                                            'alvira_utm_x',
                                                            'alvira_utm_y',
                                                            'arcus_utm_x',
                                                            'arcus_utm_y']]])
            #print("prediction: ", pred)
            #print(f"real coordinates: {ground_truth[0]}, {ground_truth[1]} --- {ground_truth[2]}, {ground_truth[3]}")

            (x1_pred, y1_pred) = p(pred[0][0], pred[0][1], inverse=True)
            (x2_pred, y2_pred) = p(pred[0][2], pred[0][3], inverse=True)

            if self.realTime:
                self.form_widget.browser.page().runJavaScript(f"addForecastedDronesPoints({y1_pred}, {x1_pred}, {y2_pred}, {x2_pred});")
            else:
                self.form_widget.browser.page().runJavaScript(f"addToPoly1({real_y}, {real_x})")
                self.form_widget.browser.page().runJavaScript(f"addToPoly2({y1_pred}, {x1_pred})")
                self.form_widget.browser.page().runJavaScript(f"addToPoly3({real_y2}, {real_x2})")

            if pred[0][3] < 5709705.966524 or pred[0][3] > 5716020.185796:
                if ground_truth[2] == 0:
                    print("zero miss!")
                else:
                    print("second miss!")
                    self.second_miss += 1
            else:
                self.form_widget.browser.page().runJavaScript(f"addToPoly4({y2_pred}, {x2_pred})")

        if not self.realTime:
            self.form_widget.browser.page().runJavaScript("showPoly1();")
            self.form_widget.browser.page().runJavaScript("showPoly2();")
            self.form_widget.browser.page().runJavaScript("showPoly3();")
            self.form_widget.browser.page().runJavaScript("showPoly4();")

            print(f"misses: {self.second_miss} --- {(self.second_miss / self.total_counter) * 100}%")

        QtWidgets.QApplication.processEvents()
        time.sleep(1)

        self.btn1.animateClick()

class WebEnginePage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print()
        #print("javaScriptConsoleMessage: ", level, message, lineNumber, sourceID)

class FormWidget(QWidget):
    def __init__(self, parent):
        super(FormWidget, self).__init__(parent)
        self.__controls()
        self.__layout()

    def __controls(self):
        html = open('html/mapview.html', 'r').read()
        self.browser = QWebEngineView()
        self.browser.setPage(WebEnginePage(self.browser))
        self.browser.setHtml(html)
        self.browser.loadFinished.connect(self.onLoadFinished)

    def onLoadFinished(self, ok):
        print("load finished!")

    def __layout(self):
        self.vbox = QVBoxLayout()
        self.hBox = QVBoxLayout()
        self.hBox.addWidget(self.browser)
        self.vbox.addLayout(self.hBox)
        self.setLayout(self.vbox)

    def ready(self, returnValue):
        print(returnValue)

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())