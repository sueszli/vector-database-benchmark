"""
Test some of the methods in the vi_quad dataset

Uses a small section of the dataset as a test
"""
import pytest
from stanza.utils.datasets.constituency import selftrain_vi_quad
from stanza.tests import *
pytestmark = [pytest.mark.pipeline, pytest.mark.travis]
SAMPLE_TEXT = '\n{"version": "1.1", "data": [{"title": "Phạm Văn Đồng", "paragraphs": [{"qas": [{"question": "Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?", "answers": [{"answer_start": 507, "text": "Lâm Bá Kiệt"}], "id": "uit_01__05272_0_1"}, {"question": "Phạm Văn Đồng giữ chức vụ gì trong bộ máy Nhà nước Cộng hòa Xã hội chủ nghĩa Việt Nam?", "answers": [{"answer_start": 60, "text": "Thủ tướng"}], "id": "uit_01__05272_0_2"}, {"question": "Giai đoạn năm 1955-1976, Phạm Văn Đồng nắm giữ chức vụ gì?", "answers": [{"answer_start": 245, "text": "Thủ tướng Chính phủ Việt Nam Dân chủ Cộng hòa"}], "id": "uit_01__05272_0_3"}], "context": "Phạm Văn Đồng (1 tháng 3 năm 1906 – 29 tháng 4 năm 2000) là Thủ tướng đầu tiên của nước Cộng hòa Xã hội chủ nghĩa Việt Nam từ năm 1976 (từ năm 1981 gọi là Chủ tịch Hội đồng Bộ trưởng) cho đến khi nghỉ hưu năm 1987. Trước đó ông từng giữ chức vụ Thủ tướng Chính phủ Việt Nam Dân chủ Cộng hòa từ năm 1955 đến năm 1976. Ông là vị Thủ tướng Việt Nam tại vị lâu nhất (1955–1987). Ông là học trò, cộng sự của Chủ tịch Hồ Chí Minh. Ông có tên gọi thân mật là Tô, đây từng là bí danh của ông. Ông còn có tên gọi là Lâm Bá Kiệt khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm (Chủ nhiệm là Hồ Học Lãm)."}, {"qas": [{"question": "Sự kiện quan trọng nào đã diễn ra vào ngày 20/7/1954?", "answers": [{"answer_start": 364, "text": "bản Hiệp định đình chỉ chiến sự ở Việt Nam, Campuchia và Lào đã được ký kết thừa nhận tôn trọng độc lập, chủ quyền, của nước Việt Nam, Lào và Campuchia"}], "id": "uit_01__05272_1_1"}, {"question": "Chức vụ mà Phạm Văn Đồng đảm nhiệm tại Hội nghị Genève về Đông Dương?", "answers": [{"answer_start": 33, "text": "Trưởng phái đoàn Chính phủ"}], "id": "uit_01__05272_1_2"}, {"question": "Hội nghị Genève về Đông Dương có tính chất như thế nào?", "answers": [{"answer_start": 262, "text": "rất căng thẳng và phức tạp"}], "id": "uit_01__05272_1_3"}]}]}]}\n'
EXPECTED = ['Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?', 'Phạm Văn Đồng giữ chức vụ gì trong bộ máy Nhà nước Cộng hòa Xã hội chủ nghĩa Việt Nam?', 'Giai đoạn năm 1955-1976, Phạm Văn Đồng nắm giữ chức vụ gì?', 'Sự kiện quan trọng nào đã diễn ra vào ngày 20/7/1954?', 'Chức vụ mà Phạm Văn Đồng đảm nhiệm tại Hội nghị Genève về Đông Dương?', 'Hội nghị Genève về Đông Dương có tính chất như thế nào?']

def test_read_file():
    if False:
        for i in range(10):
            print('nop')
    results = selftrain_vi_quad.parse_quad(SAMPLE_TEXT)
    assert results == EXPECTED