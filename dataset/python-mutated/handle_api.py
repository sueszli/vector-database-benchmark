from pathlib import Path
import json
from threading import Lock
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import httpx
from httpx import Limits
import asyncio
from utils.sql import Sql
from utils.req import reqFunc, default_header_user_agent
from utils.log import logger
path = Path(__file__).parent.absolute().joinpath('debug', 'api.db')
sql = Sql(db_path=path)
sql.newTable()
lock = Lock()
q = Queue()

def read_url() -> str:
    if False:
        while True:
            i = 10
    global q
    with open('GETAPI.json', 'r', encoding='utf8') as f:
        data = json.load(fp=f)
        for d in data:
            if not ((d.startswith('https://') or d.startswith('http://')) and '[phone]' in d):
                continue
            q.put(d)
    logger.info(f'GETAPI接口总数:{q.qsize()}')
    return q

def test():
    if False:
        for i in range(10):
            print('nop')
    while not q.empty():
        i = q.get()
        if reqFunc(i, '19820294268'):
            with lock:
                sql.update(i)

async def test2():
    while not q.empty():
        i = q.get()
        _i = i.replace('[phone]', '19820294267')
        async with httpx.AsyncClient(headers=default_header_user_agent(), timeout=100, limits=Limits(max_connections=1000, max_keepalive_connections=20), verify=False) as client:
            try:
                await client.get(_i)
                sql.update(i)
            except httpx.HTTPError as why:
                if why is None:
                    logger.exception('未知的失败')
                logger.error(f'请求失败{type(why)}{why} {i}')
            except Exception as e:
                logger.error('全局失败')
                logger.exception(e)

async def asMain():
    await asyncio.gather(*(test2() for _ in range(150)))

def save_api():
    if False:
        return 10
    '保存api到 GETAPI.json 文件'
    apis = sql.select()
    api_lst = [api for api in apis]
    with open('GETAPI.json', mode='w', encoding='utf8') as j:
        json.dump(fp=j, obj=api_lst, ensure_ascii=False)
    logger.success('写入到 GETAPI.json 成功!')

def main():
    if False:
        return 10
    read_url()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asMain())
if __name__ == '__main__':
    main()
    save_api()