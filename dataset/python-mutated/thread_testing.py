import logging
import threading
import time

def thread_function(name):
    if False:
        while True:
            i = 10
    logging.info('Thread %s: starting', name)
    time.sleep(2)
    logging.info('Thread %s: finishing', name)

def population_thread_function(population, agent, round):
    if False:
        print('Hello World!')
    logging.info(f'population: {population}, agent: {agent}, round: {round}')
    time.sleep(0.1)
    logging.info(f'Finish population: {population}, agent: {agent}, round: {round}')

def agent_thread_function(agent, delay, round):
    if False:
        return 10
    logging.info(f'Agent: {agent}, round: {round}')
    time.sleep(delay)
    threads = []
    for n in range(5):
        threads.append(threading.Thread(target=population_thread_function, args=(n, agent, round)))
        threads[-1].start()
    for (e, thread) in enumerate(threads):
        logging.info(f'Joing agent: {agent}, thread: {e}, round: {round}')
        thread.join()
    logging.info(f'Finish Agent: {agent}, round: {round}')
if __name__ == '__main__':
    format = '%(asctime)s: %(message)s'
    logging.basicConfig(format=format, level=logging.INFO, datefmt='%H:%M:%S')
    delay = [0.1, 0.1]
    for i in range(50):
        logging.info(f'Start round: {i}')
        threads = []
        for (e, agent) in enumerate(['pred', 'prey']):
            threads.append(threading.Thread(target=agent_thread_function, args=(agent, delay[e], i)))
            threads[-1].start()
        for (e, thread) in enumerate(threads):
            logging.info(f'Joing round: {i}, thread: {e}')
            thread.join()
        print('--------------------------------------------------')