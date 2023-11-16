import textdistance as td

def match(resume, job_des):
    if False:
        return 10
    j = td.jaccard.similarity(resume, job_des)
    s = td.sorensen_dice.similarity(resume, job_des)
    c = td.cosine.similarity(resume, job_des)
    o = td.overlap.normalized_similarity(resume, job_des)
    total = (j + s + c + o) / 4
    return total * 100