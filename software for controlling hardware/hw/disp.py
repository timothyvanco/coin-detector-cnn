from .hardware import Hardware

class Display:
    """Provides high level control of the demonstrator's display, in terms of showing coin amount"""

    def __init__(self, height, welcome):
        self.coin_values = {}
        self.offset = 0
        self.height = height
        self.welcome = welcome

    def set_values(self, coin_values):
        self.offset = 0
        self.coin_values = {}
        for key in coin_values:
            if coin_values[key] != 0:
                self.coin_values[key] = coin_values[key]

    def do_show(self):

        if not bool(self.coin_values):
            self.__write_welcome()
        else: 
            self.__write_values()

    def __write_values(self):

        # Display output
        strs = ['', '', '', '']
        i = -1
        for key in self.coin_values:
            i = i + 1
            line_index = (i // 2) - self.offset
            if line_index < 0 or line_index >= self.height:
                continue 
            else:
                strs[line_index] += '{0}: {1} '.format(key, self.coin_values[key])
        Hardware.disp_out(strs)

        # Scrolling on a rotating buffer principle
        lines = len(self.coin_values) // 2
        if(lines > self.height):
            self.offset += 1
            if self.offset > (lines - self.height):
                self.offset = 0

    def __write_welcome(self):

        if self.offset < 2:
            Hardware.disp_out(self.welcome)

        else:
            Hardware.disp_out(["  Insert coins  ", "  Vlozte mince  "])

        self.offset += 1
        if self.offset == 4:
            self.offset = 0
    

    
