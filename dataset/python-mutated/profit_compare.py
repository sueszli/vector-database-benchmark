import fire

class ProfitTool:

    def __init__(self, codes) -> None:
        if False:
            while True:
                i = 10
        if isinstance(codes, str):
            self.codes = [codes]
        elif isinstance(codes, list):
            self.codes = list(codes)
        else:
            raise TypeError('输入类型有误')

def main(codes):
    if False:
        while True:
            i = 10
    codes = codes.split(',')
    print(codes)
if __name__ == '__main__':
    fire.Fire(main)