"""Play a game, selecting random moves, and save what we see.

This can be used to check by hand the behaviour of a game, and also
as the basis for test cases.

Example usage:
```
playthrough --game kuhn_poker --params players=3
```
"""
from absl import app
from absl import flags
from absl import logging
from open_spiel.python.algorithms import generate_playthrough
FLAGS = flags.FLAGS
flags.DEFINE_string('game', 'kuhn_poker', "Name of the game, with optional parameters, e.g. 'kuhn_poker' or 'go(komi=4.5,board_size=19)'.")
flags.DEFINE_string('output_file', None, 'Where to write the data to.')
flags.DEFINE_list('actions', None, 'A (possibly partial) list of action choices to make.')
flags.DEFINE_string('update_path', None, 'If set, regenerates all playthroughs in the path.')
flags.DEFINE_bool('alsologtostdout', False, 'If True, the trace will be written to std-out while it is being constructed (in addition to the usual behavior).')
flags.DEFINE_integer('shard', 0, 'The shard to update.')
flags.DEFINE_integer('num_shards', 1, 'How many shards to use for updates.')

def main(unused_argv):
    if False:
        i = 10
        return i + 15
    if FLAGS.update_path:
        generate_playthrough.update_path(FLAGS.update_path, FLAGS.shard, FLAGS.num_shards)
    else:
        if not FLAGS.game:
            raise ValueError('Must specify game')
        actions = FLAGS.actions
        if actions is not None:
            actions = [int(x) for x in actions]
        text = generate_playthrough.playthrough(FLAGS.game, actions, alsologtostdout=FLAGS.alsologtostdout)
        if FLAGS.output_file:
            with open(FLAGS.output_file, 'w') as f:
                f.write(text)
        else:
            logging.info(text)
if __name__ == '__main__':
    app.run(main)