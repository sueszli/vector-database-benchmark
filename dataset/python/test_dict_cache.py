# Copyright 2015, 2016 OpenMarket Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from synapse.util.caches.dictionary_cache import DictionaryCache

from tests import unittest


class DictCacheTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.cache: DictionaryCache[str, str, str] = DictionaryCache(
            "foobar", max_entries=10
        )

    def test_simple_cache_hit_full(self) -> None:
        key = "test_simple_cache_hit_full"

        v = self.cache.get(key)
        self.assertIs(v.full, False)
        self.assertEqual(v.known_absent, set())
        self.assertEqual({}, v.value)

        seq = self.cache.sequence
        test_value = {"test": "test_simple_cache_hit_full"}
        self.cache.update(seq, key, test_value)

        c = self.cache.get(key)
        self.assertEqual(test_value, c.value)

    def test_simple_cache_hit_partial(self) -> None:
        key = "test_simple_cache_hit_partial"

        seq = self.cache.sequence
        test_value = {"test": "test_simple_cache_hit_partial"}
        self.cache.update(seq, key, test_value)

        c = self.cache.get(key, ["test"])
        self.assertEqual(test_value, c.value)

    def test_simple_cache_miss_partial(self) -> None:
        key = "test_simple_cache_miss_partial"

        seq = self.cache.sequence
        test_value = {"test": "test_simple_cache_miss_partial"}
        self.cache.update(seq, key, test_value)

        c = self.cache.get(key, ["test2"])
        self.assertEqual({}, c.value)

    def test_simple_cache_hit_miss_partial(self) -> None:
        key = "test_simple_cache_hit_miss_partial"

        seq = self.cache.sequence
        test_value = {
            "test": "test_simple_cache_hit_miss_partial",
            "test2": "test_simple_cache_hit_miss_partial2",
            "test3": "test_simple_cache_hit_miss_partial3",
        }
        self.cache.update(seq, key, test_value)

        c = self.cache.get(key, ["test2"])
        self.assertEqual({"test2": "test_simple_cache_hit_miss_partial2"}, c.value)

    def test_multi_insert(self) -> None:
        key = "test_simple_cache_hit_miss_partial"

        seq = self.cache.sequence
        test_value_1 = {"test": "test_simple_cache_hit_miss_partial"}
        self.cache.update(seq, key, test_value_1, fetched_keys={"test"})

        seq = self.cache.sequence
        test_value_2 = {"test2": "test_simple_cache_hit_miss_partial2"}
        self.cache.update(seq, key, test_value_2, fetched_keys={"test2"})

        c = self.cache.get(key, dict_keys=["test", "test2"])
        self.assertEqual(
            {
                "test": "test_simple_cache_hit_miss_partial",
                "test2": "test_simple_cache_hit_miss_partial2",
            },
            c.value,
        )
        self.assertEqual(c.full, False)

    def test_invalidation(self) -> None:
        """Test that the partial dict and full dicts get invalidated
        separately.
        """
        key = "some_key"

        seq = self.cache.sequence
        # start by populating a "full dict" entry
        self.cache.update(seq, key, {"a": "b", "c": "d"})

        # add a bunch of individual entries, also keeping the individual
        # entry for "a" warm.
        for i in range(20):
            self.cache.get(key, ["a"])
            self.cache.update(seq, f"key{i}", {"1": "2"})

        # We should have evicted the full dict...
        r = self.cache.get(key)
        self.assertFalse(r.full)
        self.assertTrue("c" not in r.value)

        # ... but kept the "a" entry that we kept querying.
        r = self.cache.get(key, dict_keys=["a"])
        self.assertFalse(r.full)
        self.assertEqual(r.value, {"a": "b"})
