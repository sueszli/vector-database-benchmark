import logging
import os
import json
import requests
from modelscope.version import __version__

class ModelTag(object):
    _URL = os.environ.get('MODEL_TAG_URL', None)
    BATCH_COMMIT_RESULT_URL = f'{_URL}/batchCommitResult'
    BATCH_REFRESH_STAGE_URL = f'{_URL}/batchRefreshStage'
    QUERY_MODEL_STAGE_URL = f'{_URL}/queryModelStage'
    HEADER = {'Content-Type': 'application/json'}
    MODEL_SKIP = 0
    MODEL_FAIL = 1
    MODEL_PASS = 2

    class ItemResult(object):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.result = 0
            self.name = ''
            self.info = ''

        def to_json(self):
            if False:
                print('Hello World!')
            return {'name': self.name, 'result': self.result, 'info': self.info}

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.job_name = ''
        self.job_id = ''
        self.model = ''
        self.sdk_version = ''
        self.image_version = ''
        self.domain = ''
        self.task = ''
        self.source = ''
        self.stage = ''
        self.item_result = []

    def _post_request(self, url, param):
        if False:
            return 10
        try:
            logging.info(url + ' query: ' + str(json.dumps(param, ensure_ascii=False)))
            res = requests.post(url=url, headers=self.HEADER, data=json.dumps(param, ensure_ascii=False).encode('utf8'))
            if res.status_code == 200:
                logging.info(f'{url} post结果: ' + res.text)
                res_json = json.loads(res.text)
                if int(res_json['errorCode']) == 200:
                    return res_json['content']
                else:
                    logging.error(res.text)
            else:
                logging.error(res.text)
        except Exception as e:
            logging.error(e)
        return None

    def batch_commit_result(self):
        if False:
            print('Hello World!')
        try:
            param = {'sdkVersion': self.sdk_version, 'imageVersion': self.image_version, 'source': self.source, 'jobName': self.job_name, 'jobId': self.job_id, 'modelList': [{'model': self.model, 'domain': self.domain, 'task': self.task, 'itemResult': self.item_result}]}
            return self._post_request(self.BATCH_COMMIT_RESULT_URL, param)
        except Exception as e:
            logging.error(e)
        return

    def batch_refresh_stage(self):
        if False:
            print('Hello World!')
        try:
            param = {'sdkVersion': self.sdk_version, 'imageVersion': self.image_version, 'source': self.source, 'stage': self.stage, 'modelList': [{'model': self.model, 'domain': self.domain, 'task': self.task}]}
            return self._post_request(self.BATCH_REFRESH_STAGE_URL, param)
        except Exception as e:
            logging.error(e)
        return

    def query_model_stage(self):
        if False:
            while True:
                i = 10
        try:
            param = {'sdkVersion': self.sdk_version, 'model': self.model, 'stage': self.stage, 'imageVersion': self.image_version}
            return self._post_request(self.QUERY_MODEL_STAGE_URL, param)
        except Exception as e:
            logging.error(e)
        return None
    '\n        model_tag = ModelTag()\n        model_tag.model = "XXX"\n        model_tag.sdk_version = "0.3.7"\n        model_tag.domain = "nlp"\n        model_tag.task = "word-segmentation"\n        item = model_tag.ItemResult()\n        item.result = model_tag.MODEL_PASS\n        item.name = "ALL"\n        item.info = ""\n        model_tag.item_result.append(item.to_json())\n    '

    def commit_ut_result(self):
        if False:
            for i in range(10):
                print('nop')
        if self._URL is not None and self._URL != '':
            self.job_name = 'UT'
            self.source = 'dev'
            self.stage = 'integration'
            self.batch_commit_result()
            self.batch_refresh_stage()

def commit_model_ut_result(model_name, ut_result):
    if False:
        for i in range(10):
            print('nop')
    model_tag = ModelTag()
    model_tag.model = model_name.replace('damo/', '')
    model_tag.sdk_version = __version__
    item = model_tag.ItemResult()
    item.result = ut_result
    item.name = 'ALL'
    item.info = ''
    model_tag.item_result.append(item.to_json())
    model_tag.commit_ut_result()