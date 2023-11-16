from seleniumbase import BaseCase
BaseCase.main(__name__, __file__)

class ChinesePdfTests(BaseCase):

    def test_chinese_pdf(self):
        if False:
            print('Hello World!')
        self.open('data:,')
        pdf = 'https://seleniumbase.io/cdn/pdf/unittest_zh.pdf'
        pdf_text = self.get_pdf_text(pdf, page=2)
        print('\n' + pdf_text)
        self.assert_pdf_text(pdf, '个测试类', page=2)
        self.assert_pdf_text(pdf, '运行单元测试')
        self.assert_pdf_text(pdf, '等待测试结束后显示所有结果')
        self.assert_pdf_text(pdf, '测试的执行跟方法的顺序没有关系')