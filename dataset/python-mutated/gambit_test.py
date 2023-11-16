"""Test that gambit export can be imported back."""
import collections
import tempfile
from absl import app
from absl.testing import absltest
from open_spiel.python.algorithms.gambit import export_gambit
import pyspiel

class GambitTest(absltest.TestCase):

    def test_gambit_export_can_be_imported(self):
        if False:
            i = 10
            return i + 15
        game_list = ['kuhn_poker', 'kuhn_poker(players=3)']
        for game_name in game_list:
            game_orig = pyspiel.load_game(game_name)
            gbt = export_gambit(game_orig)
            f = tempfile.NamedTemporaryFile('w', delete=False)
            f.write(gbt)
            f.flush()
            game_efg = pyspiel.load_game('efg_game(filename=%s)' % f.name)
            f.close()
            self._infoset_table_orig = collections.defaultdict(lambda : [])
            self._infoset_table_efg = collections.defaultdict(lambda : [])
            self._recursive_check(game_orig.new_initial_state(), game_efg.new_initial_state())
            self._check_infoset_isomorphism(self._infoset_table_orig, self._infoset_table_efg)

    def _recursive_check(self, g, h):
        if False:
            return 10
        self.assertEqual(g.current_player(), h.current_player())
        self.assertEqual(g.is_chance_node(), h.is_chance_node())
        self.assertEqual(g.is_terminal(), h.is_terminal())
        if g.is_terminal():
            self.assertEqual(g.returns(), h.returns())
            return
        if g.is_chance_node():
            self.assertEqual(g.chance_outcomes(), h.chance_outcomes())
        else:
            self.assertEqual(g.legal_actions(), h.legal_actions())
            self._infoset_table_orig[g.information_state_string()].append(g.history())
            self._infoset_table_efg[h.information_state_string()].append(h.history())
        for (a, b) in zip(g.legal_actions(), h.legal_actions()):
            self._recursive_check(g.child(a), h.child(b))

    def _check_infoset_isomorphism(self, a, b):
        if False:
            print('Hello World!')
        a_prime = []
        b_prime = []
        for vs in a.values():
            a_prime.append(sorted([str(v) for v in vs]))
        for vs in b.values():
            b_prime.append(sorted([str(v) for v in vs]))
        self.assertCountEqual(a_prime, b_prime)

def main(_):
    if False:
        print('Hello World!')
    absltest.main()
if __name__ == '__main__':
    app.run(main)