import unittest

import torch

from data_sampling import ShuffledNonoverlapSampler


class ShuffledNonoverlapSamplerTests(unittest.TestCase):
    def test_targets_are_disjoint_and_deterministic(self):
        a = ShuffledNonoverlapSampler(lo=7, end=1008, window_size=10, seed=23)
        b = ShuffledNonoverlapSampler(lo=7, end=1008, window_size=10, seed=23)
        starts_a = a.take(50)
        starts_b = b.take(50)
        torch.testing.assert_close(starts_a, starts_b)
        target_sets = [set(range(int(start) + 1, int(start) + 11))
                       for start in starts_a]
        union = set().union(*target_sets)
        self.assertEqual(len(union), 50 * 10)

    def test_resume_reconstructs_exact_suffix(self):
        full = ShuffledNonoverlapSampler(lo=0, end=1001, window_size=10, seed=9)
        prefix = full.take(17)
        state = full.state_dict()
        suffix = full.take(20)
        resumed = ShuffledNonoverlapSampler(lo=0, end=1001, window_size=10, seed=9)
        resumed.load_state_dict(state)
        torch.testing.assert_close(resumed.take(20), suffix)
        self.assertEqual(len(set(prefix.tolist()) & set(suffix.tolist())), 0)

    def test_exhaustion_and_metadata_mismatch_fail(self):
        sampler = ShuffledNonoverlapSampler(lo=0, end=102, window_size=10, seed=1)
        sampler.take(10)
        with self.assertRaisesRegex(RuntimeError, "exhausted"):
            sampler.take(1)
        state = sampler.state_dict()
        other = ShuffledNonoverlapSampler(lo=0, end=102, window_size=10, seed=2)
        with self.assertRaisesRegex(ValueError, "seed"):
            other.load_state_dict(state)

    def test_distributed_ranks_are_globally_disjoint(self):
        samplers = [
            ShuffledNonoverlapSampler(
                lo=3, end=2004, window_size=10, seed=41,
                rank=rank, world_size=4)
            for rank in range(4)
        ]
        starts = [sampler.take(40) for sampler in samplers]
        flattened = torch.cat(starts).tolist()
        self.assertEqual(len(flattened), len(set(flattened)))
        targets = [
            set(range(int(start) + 1, int(start) + 11))
            for start in flattened
        ]
        self.assertEqual(len(set().union(*targets)), len(flattened) * 10)

    def test_distributed_state_is_rank_portable_and_rewindable(self):
        original = [
            ShuffledNonoverlapSampler(
                lo=0, end=2001, window_size=10, seed=17,
                rank=rank, world_size=2)
            for rank in range(2)
        ]
        for sampler in original:
            sampler.take(7)
        shared_state = original[0].state_dict(cursor_rewind=2)
        resumed = [
            ShuffledNonoverlapSampler(
                lo=0, end=2001, window_size=10, seed=17,
                rank=rank, world_size=2)
            for rank in range(2)
        ]
        for sampler in resumed:
            sampler.load_state_dict(shared_state)
        for rank in range(2):
            replayed = resumed[rank].take(2)
            expected = ShuffledNonoverlapSampler(
                lo=0, end=2001, window_size=10, seed=17,
                rank=rank, world_size=2)
            expected.take(5)
            torch.testing.assert_close(replayed, expected.take(2))

    def test_schema_v1_resume_is_single_process_only(self):
        legacy = {
            "schema_version": 1, "lo": 0, "end": 101,
            "window_size": 10, "seed": 3, "n_windows": 10, "cursor": 4,
        }
        single = ShuffledNonoverlapSampler(
            lo=0, end=101, window_size=10, seed=3)
        single.load_state_dict(legacy)
        self.assertEqual(single.cursor, 4)
        distributed = ShuffledNonoverlapSampler(
            lo=0, end=101, window_size=10, seed=3, rank=0, world_size=2)
        with self.assertRaisesRegex(ValueError, "schema-v1"):
            distributed.load_state_dict(legacy)


if __name__ == "__main__":
    unittest.main()
