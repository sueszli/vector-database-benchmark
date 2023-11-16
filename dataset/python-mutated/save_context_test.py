"""Test for SaveContext."""
import threading
from tensorflow.python.eager import test
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options

class SaveContextTest(test.TestCase):

    def test_multi_thread(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(save_context.in_save_context())
        with self.assertRaisesRegex(ValueError, 'Not in a SaveContext'):
            save_context.get_save_options()
        options = save_options.SaveOptions(save_debug_info=True)
        with save_context.save_context(options):
            self.assertTrue(save_context.in_save_context())
            self.assertTrue(save_context.get_save_options().save_debug_info)
            entered_context_in_thread = threading.Event()
            continue_thread = threading.Event()

            def thread_fn():
                if False:
                    for i in range(10):
                        print('nop')
                self.assertFalse(save_context.in_save_context())
                with self.assertRaisesRegex(ValueError, 'Not in a SaveContext'):
                    save_context.get_save_options()
                options = save_options.SaveOptions(save_debug_info=False)
                with save_context.save_context(options):
                    self.assertTrue(save_context.in_save_context())
                    self.assertFalse(save_context.get_save_options().save_debug_info)
                    entered_context_in_thread.set()
                    continue_thread.wait()
                self.assertFalse(save_context.in_save_context())
                with self.assertRaisesRegex(ValueError, 'Not in a SaveContext'):
                    save_context.get_save_options()
            t = threading.Thread(target=thread_fn)
            t.start()
            entered_context_in_thread.wait()
            self.assertTrue(save_context.in_save_context())
            self.assertTrue(save_context.get_save_options().save_debug_info)
            continue_thread.set()
            t.join()
            self.assertTrue(save_context.in_save_context())
            self.assertTrue(save_context.get_save_options().save_debug_info)
        self.assertFalse(save_context.in_save_context())
        with self.assertRaisesRegex(ValueError, 'Not in a SaveContext'):
            save_context.get_save_options()

    def test_enter_multiple(self):
        if False:
            print('Hello World!')
        options = save_options.SaveOptions()
        with self.assertRaisesRegex(ValueError, 'Already in a SaveContext'):
            with save_context.save_context(options):
                with save_context.save_context(options):
                    pass
if __name__ == '__main__':
    test.main()