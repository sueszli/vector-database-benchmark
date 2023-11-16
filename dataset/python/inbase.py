if __name__ == '__main__':
    import os
    import django

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MultiMedia.settings')
    django.setup()
    from SearchSystem.models import Music, Images

    class_mapping = {
        0: '0',
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9'
    }

    root_path = 'D:\PythonProjects\MultiMedia\SearchSystem\static\Music\\'
    for index in range(10):
        path = root_path + class_mapping[index] + '\\'
        music_list = os.listdir(path)
        for music in music_list:
            music_record = Music(name=music, location=path + music,  tag=index)
            music_record.save()
            print('%d ok!' % index)
