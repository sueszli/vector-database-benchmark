from IPython.display import display, HTML

def Confetti():
    if False:
        print('Hello World!')
    display(HTML('\n    <script type="module">import confetti from \'https://cdn.skypack.dev/canvas-confetti\'; \n    confetti({\n    particleCount: 300, \n    angle: 90, \n    spread: 180,\n    decay: 0.9,\n    startVelocity: 40,\n    origin: {\n        x: 0.5,\n        y: 0.5\n    }\n    });\n    \n    </script>\n    '))