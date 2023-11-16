from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from MainWindow import Ui_MainWindow
from datetime import datetime
import json
import os
import sys
import requests
from urllib.parse import urlencode
OPENWEATHERMAP_API_KEY = os.environ.get('OPENWEATHERMAP_API_KEY')
'\nGet an API key from https://openweathermap.org/ to use with this\napplication.\n\n'

def from_ts_to_time_of_day(ts):
    if False:
        for i in range(10):
            print('nop')
    dt = datetime.fromtimestamp(ts)
    return dt.strftime('%I%p').lstrip('0')

class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    """
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(dict, dict)

class WeatherWorker(QRunnable):
    """
    Worker thread for weather updates.
    """
    signals = WorkerSignals()
    is_interrupted = False

    def __init__(self, location):
        if False:
            while True:
                i = 10
        super(WeatherWorker, self).__init__()
        self.location = location

    @pyqtSlot()
    def run(self):
        if False:
            return 10
        try:
            params = dict(q=self.location, appid=OPENWEATHERMAP_API_KEY)
            url = 'http://api.openweathermap.org/data/2.5/weather?%s&units=metric' % urlencode(params)
            r = requests.get(url)
            weather = json.loads(r.text)
            if weather['cod'] != 200:
                raise Exception(weather['message'])
            url = 'http://api.openweathermap.org/data/2.5/forecast?%s&units=metric' % urlencode(params)
            r = requests.get(url)
            forecast = json.loads(r.text)
            self.signals.result.emit(weather, forecast)
        except Exception as e:
            self.signals.error.emit(str(e))
        self.signals.finished.emit()

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.pushButton.pressed.connect(self.update_weather)
        self.threadpool = QThreadPool()
        self.show()

    def alert(self, message):
        if False:
            return 10
        alert = QMessageBox.warning(self, 'Warning', message)

    def update_weather(self):
        if False:
            print('Hello World!')
        worker = WeatherWorker(self.lineEdit.text())
        worker.signals.result.connect(self.weather_result)
        worker.signals.error.connect(self.alert)
        self.threadpool.start(worker)

    def weather_result(self, weather, forecasts):
        if False:
            i = 10
            return i + 15
        self.latitudeLabel.setText('%.2f °' % weather['coord']['lat'])
        self.longitudeLabel.setText('%.2f °' % weather['coord']['lon'])
        self.windLabel.setText('%.2f m/s' % weather['wind']['speed'])
        self.temperatureLabel.setText('%.1f °C' % weather['main']['temp'])
        self.pressureLabel.setText('%d' % weather['main']['pressure'])
        self.humidityLabel.setText('%d' % weather['main']['humidity'])
        self.sunriseLabel.setText(from_ts_to_time_of_day(weather['sys']['sunrise']))
        self.weatherLabel.setText('%s (%s)' % (weather['weather'][0]['main'], weather['weather'][0]['description']))
        self.set_weather_icon(self.weatherIcon, weather['weather'])
        for (n, forecast) in enumerate(forecasts['list'][:5], 1):
            getattr(self, 'forecastTime%d' % n).setText(from_ts_to_time_of_day(forecast['dt']))
            self.set_weather_icon(getattr(self, 'forecastIcon%d' % n), forecast['weather'])
            getattr(self, 'forecastTemp%d' % n).setText('%.1f °C' % forecast['main']['temp'])

    def set_weather_icon(self, label, weather):
        if False:
            while True:
                i = 10
        label.setPixmap(QPixmap(os.path.join('images', '%s.png' % weather[0]['icon'])))
if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    app.exec_()