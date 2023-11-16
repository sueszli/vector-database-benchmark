import requests, time, random
import hmac, json, base64
from bs4 import BeautifulSoup
from hashlib import sha1
import TencentYoutuyun
from PIL import Image
import uuid

def recognition_captcha(data):
    if False:
        print('Hello World!')
    ' 识别验证码 '
    file_id = str(uuid.uuid1())
    filename = 'captcha_' + file_id + '.gif'
    filename_png = 'captcha_' + file_id + '.png'
    if data is None:
        return
    data = base64.b64decode(data.encode('utf-8'))
    with open(filename, 'wb') as fb:
        fb.write(data)
    appid = 'appid'
    secret_id = 'secret_id'
    secret_key = 'secret_key'
    userid = 'userid'
    end_point = TencentYoutuyun.conf.API_YOUTU_END_POINT
    youtu = TencentYoutuyun.YouTu(appid, secret_id, secret_key, userid, end_point)
    im = Image.open(filename)
    im.save(filename_png, 'png')
    im.close()
    result = youtu.generalocr(filename_png, data_type=0, seq='')
    return result

def get_captcha(sessiona, headers):
    if False:
        print('Hello World!')
    ' 获取验证码 '
    need_cap = False
    while need_cap is not True:
        try:
            sessiona.get('https://www.zhihu.com/signin', headers=headers)
            resp2 = sessiona.get('https://www.zhihu.com/api/v3/oauth/captcha?lang=cn', headers=headers)
            need_cap = json.loads(resp2.text)['show_captcha']
            time.sleep(0.5 + random.randint(1, 9) / 10)
        except Exception:
            continue
    try:
        resp3 = sessiona.put('https://www.zhihu.com/api/v3/oauth/captcha?lang=cn', headers=headers)
        img_data = json.loads(resp3.text)['img_base64']
    except Exception:
        return
    return img_data

def create_point(point_data, confidence):
    if False:
        return 10
    ' 获得点阵 '
    points = {1: [20.5, 25.1875], 2: [45.5, 25.1875], 3: [70.5, 25.1875], 4: [95.5, 25.1875], 5: [120.5, 25.1875], 6: [145.5, 25.1875], 7: [170.5, 25.1875]}
    wi = 0
    input_points = []
    for word in point_data['items'][0]['words']:
        wi = wi + 1
        if word['confidence'] < confidence:
            try:
                input_points.append(points[wi])
            except KeyError:
                continue
    if len(input_points) > 2 or len(input_points) == 0:
        return []
    result = {}
    result['img_size'] = [200, 44]
    result['input_points'] = input_points
    result = json.dumps(result)
    print(result)
    return result

def bolting(k_low, k_hi, k3_confidence):
    if False:
        for i in range(10):
            print('nop')
    ' 筛选把握大的进行验证 '
    start = time.time()
    is_success = False
    while is_success is not True:
        points_len = 1
        angle = -20
        img_ko = []
        while points_len != 21 or angle < k_low or angle > k_hi:
            img_data = get_captcha(sessiona, headers)
            img_ko = recognition_captcha(img_data)
            try:
                points_len = len(img_ko['items'][0]['itemstring'])
                angle = img_ko['angle']
            except Exception:
                points_len = 1
                angle = -20
                continue
        input_text = create_point(img_ko, k3_confidence)
        if type(input_text) == type([]):
            continue
        data = {'input_text': input_text}
        time.sleep(4 + random.randint(1, 9) / 10)
        try:
            resp5 = sessiona.post('https://www.zhihu.com/api/v3/oauth/captcha?lang=cn', data, headers=headers)
        except Exception:
            continue
        print('angle: ' + str(angle))
        print(BeautifulSoup(resp5.content, 'html.parser'))
        print('-' * 50)
        try:
            is_success = json.loads(resp5.text)['success']
        except KeyError:
            continue
    end = time.time()
    return end - start
if __name__ == '__main__':
    sessiona = requests.Session()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0', 'authorization': 'oauth c3cef7c66a1843f8b3a9e6a1e3160e20'}
    k3_confidence = 0.71
    '\n    # 可视化数据会被保存在云端供浏览\n    # https://plot.ly/~weldon2010/4\n    # 纯属学习，并未看出"角度"范围扩大对图像识别的影响，大部分时候60s内能搞定，说明优图还是很强悍的，识别速度也非常快\n    '
    runtime_list_x = []
    runtime_list_y = []
    nn = range(1, 11)
    for y in nn:
        for x in nn:
            runtime_list_x.append(bolting(-3, 3, k3_confidence))
            print('y: ' + str(runtime_list_y))
            print('x: ' + str(runtime_list_x))
        runtime_list_y.append(runtime_list_x.copy())
        runtime_list_x = []
    print('-' * 30)
    print(runtime_list_y)
    print('-' * 30)
    import plotly
    import plotly.graph_objs as go
    plotly.tools.set_credentials_file(username='username', api_key='username')
    trace = go.Heatmap(z=runtime_list_y, x=[n for n in nn], y=[n for n in nn])
    data = [trace]
    plotly.plotly.plot(data, filename='weldon-time2-heatmap')