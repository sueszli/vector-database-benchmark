def _optimize(tagList, tagName, conversion):
    if False:
        print('Hello World!')
    newTagList = []
    for tag in tagList:
        if tag.name == tagName or tag.name == 'rawtext':
            newTagList.append(tag)
    for (i, newTag) in enumerate(newTagList[:-1]):
        if newTag.name == tagName and newTagList[i + 1].name == tagName:
            tagList.remove(newTag)
    newTagList = []
    for tag in tagList:
        if tag.name == tagName:
            newTagList.append(tag)
    for (i, newTag) in enumerate(newTagList[:-1]):
        value = conversion(newTag.parameter)
        nextValue = conversion(newTagList[i + 1].parameter)
        if value == nextValue:
            tagList.remove(newTagList[i + 1])
    while len(tagList) > 0 and tagList[-1].name == tagName:
        del tagList[-1]

def tagListOptimizer(tagList):
    if False:
        while True:
            i = 10
    oldSize = len(tagList)
    _optimize(tagList, 'fontsize', int)
    _optimize(tagList, 'fontweight', int)
    return oldSize - len(tagList)