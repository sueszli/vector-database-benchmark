PYTHON = ['3.6.10', '3.7.7', '3.8.3']
DJANGO = ['2.1.15', '2.2.12', '3.0.6', '3.1']
HEADER = '\nFROM python:3.8\n\nRUN apt install curl\nRUN curl https://pyenv.run | bash\n\nENV HOME  /root\nENV PYENV_ROOT $HOME/.pyenv\nENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH\n\nRUN eval "$(pyenv init -)"\nRUN eval "$(pyenv virtualenv-init -)"\n\nCOPY tests/env-matrix/install_env.sh /install_env.sh\nRUN chmod u+x /install_env.sh\n'.strip()

def envname(py, dj):
    if False:
        while True:
            i = 10
    py = ''.join(py.split('.')[:2])
    dj = ''.join(dj.split('.')[:2])[:2]
    return f'env-{py}-{dj}'
print(HEADER)
for py in PYTHON:
    print(f'RUN pyenv install {py}')
for d in DJANGO:
    print()
    print(f'# Django {d}')
    for p in PYTHON:
        e = envname(p, d)
        print(f'RUN /install_env.sh {p:<7} {d:<7} {e}')
print('\nCOPY ninja /ninja\nCOPY tests /tests\nCOPY docs /docs\nCOPY tests/env-matrix/run.sh /run.sh\nRUN chmod u+x /run.sh\n')
print("RUN echo 'Dependencies installed. Now running tests...' &&\\")
for d in DJANGO:
    for p in PYTHON:
        e = envname(p, d)
        print(f'    /run.sh {e}  &&\\')
print("    echo 'Done.'")