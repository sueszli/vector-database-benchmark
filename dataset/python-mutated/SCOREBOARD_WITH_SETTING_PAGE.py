import remi.gui as gui
from remi import start, App
import os
import time
import threading

class MyApp(App):

    def __init__(self, *args):
        if False:
            return 10
        res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'res')
        super(MyApp, self).__init__(*args, static_file_path={'res': res_path})

    def idle(self):
        if False:
            print('Hello World!')
        pass

    def ChangeColor(self, Text, Side, Num, BtUp, BtDn):
        if False:
            while True:
                i = 10
        self.In = 1
        with self.update_lock:
            Text.set_text(str(Num))
            Side.style['background-color'] = 'red'
            BtUp.attributes['class'] = 'up80red'
            BtDn.attributes['class'] = 'dn80red'
        time.sleep(3)
        with self.update_lock:
            Side.style['background-color'] = 'green'
            BtUp.attributes['class'] = 'up80'
            BtDn.attributes['class'] = 'dn80'
            self.LeftNum = 0
            self.RightNum = 0
        self.In = 0
        self.check_score()

    def main(self):
        if False:
            print('Hello World!')
        self.In = 0
        self.LeftNum = 0
        self.RightNum = 0
        self.MatchNum = 5
        self.LeftMatchNum = 0
        self.RightMatchNum = 0
        self.Name1 = 'LEFT'
        self.Name2 = 'RIGHT'
        widMenu = gui.Container(width=480, height=610, layout_orientation=gui.Container.LAYOUT_VERTICAL, style={'margin': '0px auto', 'background': 'black'})
        self.lblMenu = gui.Label('SCOREBOARD', width='100%', height='45px', style={'margin': '0px 0px 0px', 'padding-top': '10px', 'font-size': '40px', 'font-weight': 'bold', 'color': 'green', 'line-height': '45px', 'text-align': 'center'})
        self.lblMenu2 = gui.Label('Setup players name:', width='100%', height='45px', style={'margin': '0px 0px 0px', 'padding-top': '10px', 'font-size': '30px', 'font-weight': 'bold', 'line-height': '45px', 'text-align': 'left'})
        self.lblName1 = gui.Label('PLAYER 1 NAME:', width='100%', height='35px', style={'margin': '0px 0px 0px', 'padding-top': '20px', 'font-size': '20px', 'line-height': '25px', 'text-align': 'left'})
        self.txtName1 = gui.TextInput(width='96%', height='35px', style={'margin': '0px auto', 'padding-top': '20px', 'padding-left': '5px', 'font-size': '30px', 'line-height': '20px', 'text-align': 'left', 'border': '1px solid white', 'background': 'black'})
        self.txtName1.set_text('P1')
        self.lblName2 = gui.Label('PLAYER 2 NAME:', width='100%', height='35px', style={'margin': '0px 0px 0px', 'padding-top': '20px', 'font-size': '20px', 'line-height': '25px', 'text-align': 'left'})
        self.txtName2 = gui.TextInput(width='96%', height='35px', style={'margin': '0px auto', 'padding-top': '20px', 'padding-left': '5px', 'font-size': '30px', 'line-height': '20px', 'text-align': 'left', 'border': '1px solid white', 'background': 'black'})
        self.txtName2.set_text('P2')
        btMenu = gui.Button('START', width='40%', height='40px', style={'margin': '50px 20% 20px', 'font-size': '30px', 'line-height': '30px', 'text-align': 'center'})
        widMenu.append([self.lblMenu, self.lblMenu2, self.lblName1, self.txtName1, self.lblName2, self.txtName2, btMenu])
        btMenu.onclick.connect(self.on_button_pressed_menu)
        wid = gui.Container(width=480, height=610, style={'margin': '0px auto', 'background': 'black'})
        self.lbl = gui.Label('SCOREBOARD', width='100%', height='35px', style={'margin': '0px 0px 0px', 'padding-top': '10px', 'font-size': '30px', 'line-height': '35px', 'text-align': 'center'})
        wid1 = gui.Container(width='100%', height=600, layout_orientation=gui.Container.LAYOUT_HORIZONTAL, style={'background': 'black'})
        self.wid2 = gui.Container(width=230, height=350, margin='5px', style={'background': 'green'})
        self.wid3 = gui.Container(width=230, height=350, margin='5px', style={'background': 'green'})
        self.lblLeftName = gui.Label(self.Name1, width='95%', height='60px', style={'margin': '20px 2px 0px', 'font-size': '40px', 'line-height': '60px', 'text-align': 'center', 'overflow': 'hidden'})
        self.lblLeftNum = gui.Label(str(self.LeftNum), width='100%', height='130px', style={'margin': '0px 0px 10px', 'font-size': '140px', 'line-height': '130px', 'text-align': 'center'})
        self.btLeftPlus = gui.Button('', width='80px', height='80px', style={'margin': '0px 10px 20px', 'font-size': '50px', 'line-height': '50px', 'text-align': 'center'})
        self.btLeftPlus.attributes['class'] = 'up80'
        self.btLeftMinus = gui.Button('', width='80px', height='80px', style={'margin': '0px 10px 20px', 'font-size': '50px', 'line-height': '50px', 'text-align': 'center'})
        self.btLeftMinus.attributes['class'] = 'dn80'
        lblLeftMatch = gui.Label('MATCHES WON:', width=150, height='30px', style={'margin': '0px 5px', 'font-size': '20px', 'line-height': '30px', 'text-align': 'left', 'display': 'inline'})
        self.lblLeftMatches = gui.Label(str(self.LeftMatchNum), width=30, height='30px', style={'margin': '0px 5px', 'font-size': '20px', 'line-height': '30px', 'text-align': 'left', 'display': 'inline'})
        self.lblRightName = gui.Label(self.Name2, width='95%', height='60px', style={'margin': '20px 2px 0px', 'font-size': '40px', 'line-height': '60px', 'text-align': 'center', 'overflow': 'hidden'})
        self.lblRightNum = gui.Label(str(self.LeftNum), width='100%', height='130px', style={'margin': '0px 0px 10px', 'font-size': '140px', 'line-height': '130px', 'text-align': 'center'})
        self.btRightPlus = gui.Button('', width='80px', height='80px', style={'margin': '0px 10px 20px', 'font-size': '50px', 'line-height': '50px', 'text-align': 'center'})
        self.btRightPlus.attributes['class'] = 'up80'
        self.btRightMinus = gui.Button('', width='80px', height='80px', style={'margin': '0px 10px 20px', 'font-size': '50px', 'line-height': '50px', 'text-align': 'center'})
        self.btRightMinus.attributes['class'] = 'dn80'
        lblRightMatch = gui.Label('MATCHES WON:', width=150, height='30px', style={'margin': '0px 5px', 'font-size': '20px', 'line-height': '30px', 'text-align': 'left', 'display': 'inline'})
        self.lblRightMatches = gui.Label(str(self.RightMatchNum), width=30, height='30px', style={'margin': '0px 5px', 'font-size': '20px', 'line-height': '30px', 'text-align': 'left', 'display': 'inline'})
        self.wid2.append([self.lblLeftName, self.lblLeftNum, self.btLeftPlus, self.btLeftMinus, lblLeftMatch, self.lblLeftMatches])
        self.wid3.append([self.lblRightName, self.lblRightNum, self.btRightPlus, self.btRightMinus, lblRightMatch, self.lblRightMatches])
        wid1.append(self.wid2)
        wid1.append(self.wid3)
        lblMatch = gui.Label('GAMES FOR MATCH:', width='50%', height='50px', style={'margin': '15px 2px 0px 10px', 'font-size': '25px', 'line-height': '35px', 'text-align': 'center'})
        self.lblMatches = gui.Label(str(self.MatchNum), width='8%', height='50px', style={'margin': '15px 2px 0px', 'font-size': '25px', 'line-height': '35px', 'text-align': 'center'})
        btMatchPlus = gui.Button('', width='50px', height='50px', style={'margin': '5px 2px 0px 20px', 'font-size': '30px', 'line-height': '30px', 'text-align': 'center'})
        btMatchPlus.attributes['class'] = 'up50'
        btMatchMinus = gui.Button('', width='50px', height='50px', style={'margin': '5px 2px', 'font-size': '30px', 'line-height': '30px', 'text-align': 'center'})
        btMatchMinus.attributes['class'] = 'dn50'
        wid1.append([lblMatch, btMatchPlus, self.lblMatches, btMatchMinus])
        btReset = gui.Button('RESET SCORE', width='50%', height='35px', style={'margin': '10px 25% 10px', 'font-size': '25px', 'line-height': '30px', 'text-align': 'center'})
        wid1.append(btReset)
        btResetMatch = gui.Button('RESET MATCH', width='50%', height='35px', style={'margin': '10px 25% 10px', 'font-size': '25px', 'line-height': '30px', 'text-align': 'center'})
        wid1.append(btResetMatch)
        btSetting = gui.Button('SETTINGS', width='50%', height='35px', style={'margin': '10px 25% 20px', 'font-size': '25px', 'line-height': '30px', 'text-align': 'center'})
        wid1.append(btSetting)
        self.btLeftPlus.onclick.connect(self.on_button_pressed_plus, 'LT')
        self.btLeftMinus.onclick.connect(self.on_button_pressed_minus, 'LT')
        self.btRightPlus.onclick.connect(self.on_button_pressed_plus, 'RT')
        self.btRightMinus.onclick.connect(self.on_button_pressed_minus, 'RT')
        btMatchPlus.onclick.connect(self.on_button_pressed_match, 'PLUS')
        btMatchMinus.onclick.connect(self.on_button_pressed_match, 'MINUS')
        btReset.onclick.connect(self.on_button_pressed_reset)
        btResetMatch.onclick.connect(self.on_button_pressed_reset_match)
        btSetting.onclick.connect(self.on_button_setting)
        wid.append(self.lbl)
        wid.append(wid1)
        self.wid = wid
        self.widMenu = widMenu
        return self.widMenu

    @staticmethod
    def name_length(Name):
        if False:
            return 10
        if len(Name) <= 6:
            return (Name, 40)
        elif len(Name) <= 8:
            return (Name, 30)
        elif len(Name) <= 10:
            return (Name, 22)
        else:
            Name = Name[:14]
            return (Name, 22)

    def on_button_pressed_menu(self, emitter):
        if False:
            for i in range(10):
                print('nop')
        Name = self.txtName1.get_text()
        (Name, FntSize) = MyApp.name_length(Name)
        FntSize = str(FntSize) + 'px'
        self.lblLeftName.style['font-size'] = FntSize
        self.lblLeftName.set_text(Name)
        Name = self.txtName2.get_text()
        (Name, FntSize) = MyApp.name_length(Name)
        FntSize = str(FntSize) + 'px'
        self.lblRightName.style['font-size'] = FntSize
        self.lblRightName.set_text(Name)
        self.set_root_widget(self.wid)

    def on_button_setting(self, emitter):
        if False:
            i = 10
            return i + 15
        self.set_root_widget(self.widMenu)

    def check_score(self):
        if False:
            i = 10
            return i + 15
        if self.LeftNum < self.MatchNum and self.RightNum < self.MatchNum:
            self.lblLeftNum.set_text(str(self.LeftNum))
            self.lblRightNum.set_text(str(self.RightNum))
            self.lblLeftMatches.set_text(str(self.LeftMatchNum))
            self.lblRightMatches.set_text(str(self.RightMatchNum))
            self.lblMatches.set_text(str(self.MatchNum))
        if self.LeftNum < self.MatchNum - 1:
            self.wid2.style['background-color'] = 'green'
            self.btLeftPlus.attributes['class'] = 'up80'
            self.btLeftMinus.attributes['class'] = 'dn80'
        if self.RightNum < self.MatchNum - 1:
            self.wid3.style['background-color'] = 'green'
            self.btRightPlus.attributes['class'] = 'up80'
            self.btRightMinus.attributes['class'] = 'dn80'
        if self.LeftNum == self.MatchNum - 1:
            self.wid2.style['background-color'] = 'orange'
            self.btLeftPlus.attributes['class'] = 'up80org'
            self.btLeftMinus.attributes['class'] = 'dn80org'
        if self.RightNum == self.MatchNum - 1:
            self.wid3.style['background-color'] = 'orange'
            self.btRightPlus.attributes['class'] = 'up80org'
            self.btRightMinus.attributes['class'] = 'dn80org'
        if self.LeftNum >= self.MatchNum:
            Side = [self.lblLeftNum, self.wid2, self.LeftNum, self.btLeftPlus, self.btLeftMinus]
            t = threading.Thread(target=self.ChangeColor, args=Side)
            t.start()
            self.LeftMatchNum = self.LeftMatchNum + 1
        elif self.RightNum >= self.MatchNum:
            Side = [self.lblRightNum, self.wid3, self.RightNum, self.btRightPlus, self.btRightMinus]
            t = threading.Thread(target=self.ChangeColor, args=Side)
            t.start()
            self.RightMatchNum = self.RightMatchNum + 1

    def on_button_pressed_plus(self, emitter, Side):
        if False:
            while True:
                i = 10
        if not self.In:
            if Side == 'LT':
                if self.LeftNum < self.MatchNum:
                    self.LeftNum = self.LeftNum + 1
            elif Side == 'RT':
                if self.RightNum < self.MatchNum:
                    self.RightNum = self.RightNum + 1
            self.check_score()

    def on_button_pressed_minus(self, emitter, Side):
        if False:
            while True:
                i = 10
        if not self.In:
            if Side == 'LT':
                if self.LeftNum != 0:
                    self.LeftNum = self.LeftNum - 1
            elif Side == 'RT':
                if self.RightNum != 0:
                    self.RightNum = self.RightNum - 1
            self.check_score()

    def on_button_pressed_match(self, emitter, Side):
        if False:
            for i in range(10):
                print('nop')
        if not self.In:
            if Side == 'PLUS':
                self.MatchNum = self.MatchNum + 1
            elif Side == 'MINUS':
                if self.MatchNum > 1:
                    if self.MatchNum - 1 <= self.LeftNum:
                        self.LeftNum = self.LeftNum - 1
                    if self.MatchNum - 1 <= self.RightNum:
                        self.RightNum = self.RightNum - 1
                    self.MatchNum = self.MatchNum - 1
            self.check_score()

    def on_button_pressed_reset(self, emitter):
        if False:
            for i in range(10):
                print('nop')
        if not self.In:
            self.LeftNum = 0
            self.RightNum = 0
            self.check_score()

    def on_button_pressed_reset_match(self, emitter):
        if False:
            i = 10
            return i + 15
        if not self.In:
            self.LeftMatchNum = 0
            self.RightMatchNum = 0
            self.check_score()
if __name__ == '__main__':
    start(MyApp, address='', port=8081, multiple_instance=False, enable_file_cache=True, update_interval=0.1, start_browser=True)