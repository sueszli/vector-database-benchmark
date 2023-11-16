import time
import random
import base64
import hashlib
import requests
from urllib.parse import urlencode
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def random_str():
    if False:
        return 10
    '得到随机字符串nonce_str'
    str = 'abcdefghijklmnopqrstuvwxyz'
    r = ''
    for i in range(15):
        index = random.randint(0, 25)
        r += str[index]
    return r

def image(name):
    if False:
        for i in range(10):
            print('nop')
    with open(name, 'rb') as f:
        content = f.read()
    return base64.b64encode(content)

def get_params(img):
    if False:
        i = 10
        return i + 15
    '组织接口请求的参数形式，并且计算sign接口鉴权信息，\n    最终返回接口请求所需要的参数字典'
    params = {'app_id': '1106860829', 'time_stamp': str(int(time.time())), 'nonce_str': random_str(), 'image': img, 'mode': '0'}
    sort_dict = sorted(params.items(), key=lambda item: item[0], reverse=False)
    sort_dict.append(('app_key', 'P8Gt8nxi6k8vLKbS'))
    rawtext = urlencode(sort_dict).encode()
    sha = hashlib.md5()
    sha.update(rawtext)
    md5text = sha.hexdigest().upper()
    params['sign'] = md5text
    return params

def access_api(img):
    if False:
        return 10
    frame = cv2.imread(img)
    nparry_encode = cv2.imencode('.jpg', frame)[1]
    data_encode = np.array(nparry_encode)
    img_encode = base64.b64encode(data_encode)
    url = 'https://api.ai.qq.com/fcgi-bin/face/face_detectface'
    res = requests.post(url, get_params(img_encode)).json()
    if res['ret'] == 0:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        for obj in res['data']['face_list']:
            img_width = res['data']['image_width']
            img_height = res['data']['image_height']
            x = obj['x']
            y = obj['y']
            w = obj['width']
            h = obj['height']
            if obj['glass'] == 1:
                glass = '有'
            else:
                glass = '无'
            if obj['gender'] >= 70:
                gender = '男'
            elif 50 <= obj['gender'] < 70:
                gender = '娘'
            elif obj['gender'] < 30:
                gender = '女'
            else:
                gender = '女汉子'
            if 90 < obj['expression'] <= 100:
                expression = '一笑倾城'
            elif 80 < obj['expression'] <= 90:
                expression = '心花怒放'
            elif 70 < obj['expression'] <= 80:
                expression = '兴高采烈'
            elif 60 < obj['expression'] <= 70:
                expression = '眉开眼笑'
            elif 50 < obj['expression'] <= 60:
                expression = '喜上眉梢'
            elif 40 < obj['expression'] <= 50:
                expression = '喜气洋洋'
            elif 30 < obj['expression'] <= 40:
                expression = '笑逐颜开'
            elif 20 < obj['expression'] <= 30:
                expression = '似笑非笑'
            elif 10 < obj['expression'] <= 20:
                expression = '半嗔半喜'
            elif 0 <= obj['expression'] <= 10:
                expression = '黯然伤神'
            delt = h // 5
            if len(res['data']['face_list']) > 1:
                font = ImageFont.truetype('yahei.ttf', w // 8, encoding='utf-8')
                draw.text((x + 10, y + 10), '性别 :' + gender, (76, 176, 80), font=font)
                draw.text((x + 10, y + 10 + delt * 1), '年龄 :' + str(obj['age']), (76, 176, 80), font=font)
                draw.text((x + 10, y + 10 + delt * 2), '表情 :' + expression, (76, 176, 80), font=font)
                draw.text((x + 10, y + 10 + delt * 3), '魅力 :' + str(obj['beauty']), (76, 176, 80), font=font)
                draw.text((x + 10, y + 10 + delt * 4), '眼镜 :' + glass, (76, 176, 80), font=font)
            elif img_width - x - w < 170:
                font = ImageFont.truetype('yahei.ttf', w // 8, encoding='utf-8')
                draw.text((x + 10, y + 10), '性别 :' + gender, (76, 176, 80), font=font)
                draw.text((x + 10, y + 10 + delt * 1), '年龄 :' + str(obj['age']), (76, 176, 80), font=font)
                draw.text((x + 10, y + 10 + delt * 2), '表情 :' + expression, (76, 176, 80), font=font)
                draw.text((x + 10, y + 10 + delt * 3), '魅力 :' + str(obj['beauty']), (76, 176, 80), font=font)
                draw.text((x + 10, y + 10 + delt * 4), '眼镜 :' + glass, (76, 176, 80), font=font)
            else:
                font = ImageFont.truetype('yahei.ttf', 20, encoding='utf-8')
                draw.text((x + w + 10, y + 10), '性别 :' + gender, (76, 176, 80), font=font)
                draw.text((x + w + 10, y + 10 + delt * 1), '年龄 :' + str(obj['age']), (76, 176, 80), font=font)
                draw.text((x + w + 10, y + 10 + delt * 2), '表情 :' + expression, (76, 176, 80), font=font)
                draw.text((x + w + 10, y + 10 + delt * 3), '魅力 :' + str(obj['beauty']), (76, 176, 80), font=font)
                draw.text((x + w + 10, y + 10 + delt * 4), '眼镜 :' + glass, (76, 176, 80), font=font)
            draw.rectangle((x, y, x + w, y + h), outline='#4CB050')
            cv2img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2.imwrite('faces/{}'.format(os.path.basename(img)), cv2img)
        return 'success'
    else:
        return 'fail'