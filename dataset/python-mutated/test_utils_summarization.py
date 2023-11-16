import unittest
import numpy as np
import torch
from .utils_summarization import build_mask, compute_token_type_ids, process_story, truncate_or_pad

class SummarizationDataProcessingTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.block_size = 10

    def test_fit_to_block_sequence_too_small(self):
        if False:
            print('Hello World!')
        'Pad the sequence with 0 if the sequence is smaller than the block size.'
        sequence = [1, 2, 3, 4]
        expected_output = [1, 2, 3, 4, 0, 0, 0, 0, 0, 0]
        self.assertEqual(truncate_or_pad(sequence, self.block_size, 0), expected_output)

    def test_fit_to_block_sequence_fit_exactly(self):
        if False:
            while True:
                i = 10
        'Do nothing if the sequence is the right size.'
        sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        expected_output = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(truncate_or_pad(sequence, self.block_size, 0), expected_output)

    def test_fit_to_block_sequence_too_big(self):
        if False:
            for i in range(10):
                print('nop')
        'Truncate the sequence if it is too long.'
        sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        expected_output = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(truncate_or_pad(sequence, self.block_size, 0), expected_output)

    def test_process_story_no_highlights(self):
        if False:
            return 10
        'Processing a story with no highlights returns an empty list for the summary.'
        raw_story = 'It was the year of Our Lord one thousand seven hundred and\n        seventy-five.\n\nSpiritual revelations were conceded to England at that\n        favoured period, as at this.'
        (_, summary_lines) = process_story(raw_story)
        self.assertEqual(summary_lines, [])

    def test_process_empty_story(self):
        if False:
            return 10
        'An empty story returns an empty collection of lines.'
        raw_story = ''
        (story_lines, summary_lines) = process_story(raw_story)
        self.assertEqual(story_lines, [])
        self.assertEqual(summary_lines, [])

    def test_process_story_with_missing_period(self):
        if False:
            for i in range(10):
                print('nop')
        raw_story = 'It was the year of Our Lord one thousand seven hundred and seventy-five\n\nSpiritual revelations were conceded to England at that favoured period, as at this.\n@highlight\n\nIt was the best of times'
        (story_lines, summary_lines) = process_story(raw_story)
        expected_story_lines = ['It was the year of Our Lord one thousand seven hundred and seventy-five.', 'Spiritual revelations were conceded to England at that favoured period, as at this.']
        self.assertEqual(expected_story_lines, story_lines)
        expected_summary_lines = ['It was the best of times.']
        self.assertEqual(expected_summary_lines, summary_lines)

    def test_build_mask_no_padding(self):
        if False:
            while True:
                i = 10
        sequence = torch.tensor([1, 2, 3, 4])
        expected = torch.tensor([1, 1, 1, 1])
        np.testing.assert_array_equal(build_mask(sequence, 0).numpy(), expected.numpy())

    def test_build_mask(self):
        if False:
            return 10
        sequence = torch.tensor([1, 2, 3, 4, 23, 23, 23])
        expected = torch.tensor([1, 1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(build_mask(sequence, 23).numpy(), expected.numpy())

    def test_build_mask_with_padding_equal_to_one(self):
        if False:
            return 10
        sequence = torch.tensor([8, 2, 3, 4, 1, 1, 1])
        expected = torch.tensor([1, 1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(build_mask(sequence, 1).numpy(), expected.numpy())

    def test_compute_token_type_ids(self):
        if False:
            print('Hello World!')
        separator = 101
        batch = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 101, 5, 6], [1, 101, 3, 4, 101, 6]])
        expected = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 1, 1]])
        result = compute_token_type_ids(batch, separator)
        np.testing.assert_array_equal(result, expected)