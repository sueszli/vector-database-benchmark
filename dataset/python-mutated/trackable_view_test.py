"""Tests for the trackable view."""
from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.eager import test
from tensorflow.python.trackable import base

class TrackableViewTest(test.TestCase):

    def test_children(self):
        if False:
            return 10
        root = base.Trackable()
        leaf = base.Trackable()
        root._track_trackable(leaf, name='leaf')
        ((current_name, current_dependency),) = trackable_view.TrackableView.children(root).items()
        self.assertIs(leaf, current_dependency)
        self.assertEqual('leaf', current_name)

    def test_descendants(self):
        if False:
            i = 10
            return i + 15
        root = base.Trackable()
        leaf = base.Trackable()
        root._track_trackable(leaf, name='leaf')
        descendants = trackable_view.TrackableView(root).descendants()
        self.assertIs(2, len(descendants))
        self.assertIs(root, descendants[0])
        self.assertIs(leaf, descendants[1])
if __name__ == '__main__':
    test.main()