"""Targeted correctness checks for the MLX runner's changed paths."""

import unittest
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.qwen3 import Model, ModelArgs

from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_cache import (
    ContiguousAttentionKVCache,
    PoolBackedAttentionKVCache,
)
from sglang.srt.hardware_backend.mlx.kv_cache.attention_kv_pool import (
    MlxAttentionKVPool,
    consecutive_slots,
)
from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner


class MLXFastPathTests(unittest.TestCase):
    def test_only_ordered_consecutive_slots_become_slices(self):
        self.assertEqual(consecutive_slots([4]), slice(4, 5))
        self.assertEqual(consecutive_slots([4, 5, 6]), slice(4, 7))
        for slots in ([], [4, 6, 7], [5, 4], [5, 5]):
            self.assertIsNone(consecutive_slots(slots))

    def test_pool_slice_and_index_forward_are_equivalent_with_truncated_prefix(self):
        pool = MlxAttentionKVPool(12, 1, 2, 4, dtype=mx.float32)
        initial = mx.arange(24, dtype=mx.float32).reshape(3, 2, 4)
        pool.set_kv(0, slice(3, 6), initial, initial + 100)
        mx.eval(*pool.all_buffers())
        new = mx.ones((1, 2, 1, 4))
        for offset in (3, 2, 0):
            sliced = PoolBackedAttentionKVCache(pool, 0, slice(3, 6), offset)
            indexed = PoolBackedAttentionKVCache(pool, 0, mx.array([3, 4, 5]), offset)
            left = sliced.update_and_fetch(new, new + 100)
            right = indexed.update_and_fetch(new, new + 100)
            for a, b in zip(left, right):
                self.assertTrue(mx.array_equal(a, b).item())
            self.assertEqual(sliced.offset, offset + 1)

    def test_context_sized_cache_grows_without_losing_the_prefix(self):
        runner = MlxModelRunner.__new__(MlxModelRunner)
        runner._max_seq_len = 4096
        runner._optimize_kv_layout = True
        runner._cache_pool = []
        runner._cache_layout = SimpleNamespace(
            has_auxiliary_state=False, has_sliding_window_layers=False,
            num_layers=1, attention_layer_indices=[0], window_size=lambda _: None,
        )
        cache = runner._acquire_cache(12)[0]
        self.assertEqual(cache.max_seq_len, 128)
        keys = mx.arange(12 * 2 * 4).reshape((1, 2, 12, 4))
        cache.update_and_fetch(keys, keys + 1000)
        tail = mx.zeros((1, 2, 140, 4), dtype=keys.dtype)
        merged, values = cache.update_and_fetch(tail, tail)
        self.assertEqual(cache.offset, 152)
        self.assertEqual(cache.max_seq_len, 256)
        self.assertTrue(mx.array_equal(merged[:, :, :12], keys).item())
        self.assertTrue(mx.array_equal(values[:, :, :12], keys + 1000).item())
        runner._release_cache([cache])
        self.assertIs(runner._acquire_cache(12)[0], cache)
        self.assertEqual(cache.offset, 0)

    def test_adopted_cache_matches_copy_path_after_decode_and_reset(self):
        reference = ContiguousAttentionKVCache(max_seq_len=128)
        compact = ContiguousAttentionKVCache(max_seq_len=128)
        initial = mx.arange(48, dtype=mx.float32).reshape((1, 2, 6, 4))
        reference.update_and_fetch(initial, initial + 100)
        compact.adopt_compact(initial, initial + 100)
        self.assertEqual(compact.offset, 6)
        for increment in (4, 9):
            step = mx.full((1, 2, 1, 4), increment, dtype=mx.float32)
            reference.write_token(step, step + 100)
            compact.write_token(step, step + 100)
            for left, right in zip(reference.get_kv(), compact.get_kv()):
                self.assertTrue(mx.array_equal(left, right).item())
        compact.adopt_compact(initial, initial + 100)
        compact.reset()
        fresh = mx.full((1, 2, 2, 4), 99, dtype=mx.float32)
        keys, values = compact.update_and_fetch(fresh, fresh + 100)
        self.assertTrue(mx.array_equal(keys, fresh).item())
        self.assertTrue(mx.array_equal(values, fresh + 100).item())

    def test_direct_layer_writes_match_original_stacking_for_both_slot_layouts(self):
        data = mx.arange(40, dtype=mx.float32).reshape((1, 2, 5, 4))
        cache = [ContiguousAttentionKVCache(max_seq_len=8) for _ in range(2)]
        for index, layer in enumerate(cache):
            keys = data + 100 * index
            layer.update_and_fetch(keys, keys + 200)
        for ids in ([1, 2, 3], [4, 1, 7]):
            snapshots = []
            for optimize in (False, True):
                runner = MlxModelRunner.__new__(MlxModelRunner)
                runner._optimize_kv_layout = optimize
                runner._cache_layout = SimpleNamespace(full_attention_layer_indices=[0, 1])
                runner._attention_kv_pool = MlxAttentionKVPool(8, 2, 2, 4, dtype=mx.float32)
                runner._sync_new_kv_to_pool(cache, 1, ids)
                snapshots.append(runner._attention_kv_pool.all_buffers())
            for left, right in zip(*snapshots):
                self.assertTrue(mx.array_equal(left, right).item())

    def test_coalesced_writes_preserve_disjoint_suffixes_and_existing_prefix(self):
        first, second = [], []
        for index in range(2):
            for caches, offset in ((first, index * 100), (second, index * 100 + 25)):
                keys = mx.arange(40, dtype=mx.float32).reshape(1, 2, 5, 4) + offset
                cache = ContiguousAttentionKVCache(max_seq_len=8)
                cache.adopt_compact(keys, keys + 500)
                caches.append(cache)
        writes = [(first, 2, [3, 5]), (second, 1, [2, 7, 8])]
        snapshots = []
        for coalesced in (False, True):
            runner = MlxModelRunner.__new__(MlxModelRunner)
            runner._optimize_kv_layout = True
            runner._cache_layout = SimpleNamespace(full_attention_layer_indices=[0, 1])
            pool = MlxAttentionKVPool(10, 2, 2, 4, dtype=mx.float32)
            for layer in range(2):
                prefix = mx.full((1, 2, 4), 99 + layer, dtype=mx.float32)
                pool.set_kv(layer, slice(1, 2), prefix, prefix + 250)
            runner._attention_kv_pool = pool
            if coalesced:
                runner.commit_prefill_pool_writes(writes)
            else:
                for write in writes:
                    runner._sync_new_kv_to_pool(*write)
            snapshots.append(pool.all_buffers())
        for sequential, combined in zip(*snapshots):
            self.assertTrue(mx.array_equal(sequential, combined).item())

    def test_projecting_only_final_qwen3_position_preserves_the_distribution(self):
        input_ids = mx.array([[10, 17, 20, 9]], dtype=mx.int32)
        for tied in (False, True):
            with self.subTest(tied=tied):
                mx.random.seed(21)
                small = Model(ModelArgs(
                    model_type="qwen3", hidden_size=64, num_hidden_layers=2,
                    intermediate_size=96, num_attention_heads=4, rms_norm_eps=1e-6,
                    vocab_size=128, num_key_value_heads=2, max_position_embeddings=128,
                    head_dim=16, tie_word_embeddings=tied, rope_theta=10000.0,
                ))
                head = small.model.embed_tokens.as_linear if tied else small.lm_head
                expected = small(input_ids)[:, -1:, :]
                observed = head(small.model(input_ids)[:, -1:, :])
                self.assertTrue(mx.allclose(expected, observed, rtol=1e-5, atol=1e-5).item())
                self.assertTrue(
                    mx.array_equal(mx.argmax(expected, -1), mx.argmax(observed, -1)).item()
                )


if __name__ == "__main__":
    unittest.main()
