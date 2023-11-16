from cloudinary import Search

class SearchFolders(Search):
    FOLDERS = 'folders'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(SearchFolders, self).__init__()
        self.endpoint(self.FOLDERS)