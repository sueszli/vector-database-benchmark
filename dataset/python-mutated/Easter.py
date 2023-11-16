from robot.api import logger

def none_shall_pass(who):
    if False:
        while True:
            i = 10
    if who is not None:
        raise AssertionError('None shall pass!')
    logger.info('<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/zKhEw7nD9C4?autoplay=1" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>', html=True)