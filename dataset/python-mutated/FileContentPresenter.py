from mage_ai.api.presenters.BasePresenter import BasePresenter

class FileContentPresenter(BasePresenter):
    default_attributes = ['content', 'name', 'path']

    def present(self, **kwargs):
        if False:
            while True:
                i = 10
        return self.model.to_dict(include_content=True)