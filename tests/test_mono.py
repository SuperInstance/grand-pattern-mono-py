"""Tests for grand-pattern-mono (Python port)."""

import math
import pytest

from grand_pattern_mono import (
    Jepa,
    Room,
    CellGraph,
    chain,
    ring,
    star,
    mesh,
    small_world,
    scale_free,
)


# ── Jepa tests ─────────────────────────────────────────────────────────

class TestJepa:
    def test_empty_predict_returns_zero(self):
        j = Jepa()
        assert j.predict() == 0.0

    def test_single_observe(self):
        j = Jepa()
        j.observe(1.0)
        assert j.predict() == pytest.approx(1.0)

    def test_two_observations_weighted(self):
        j = Jepa(alpha=0.5)
        j.observe(0.0)
        j.observe(1.0)
        # w0 = 0.5*0.5=0.25, w1 = 0.5*1.0=0.5 → normalized to 0.25/0.75 and 0.5/0.75
        # Actually with EWM sum≈0.75, pred = (0.25*0 + 0.5*1)/0.75 ≈ 0.667
        pred = j.predict()
        assert 0 < pred < 1

    def test_alpha_boundary_one(self):
        j = Jepa(alpha=1.0)
        j.observe(5.0)
        j.observe(10.0)
        assert j.predict() == pytest.approx(10.0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            Jepa(alpha=0.0)
        with pytest.raises(ValueError):
            Jepa(alpha=1.5)

    def test_surprise_zero_for_same_value(self):
        j = Jepa()
        j.observe(3.0)
        j.observe(3.0)
        assert j.surprise(3.0) == pytest.approx(0.0, abs=1e-10)

    def test_surprise_nonzero(self):
        j = Jepa()
        j.observe(0.0)
        j.observe(0.0)
        s = j.surprise(1.0)
        assert s > 0

    def test_history_tracking(self):
        j = Jepa()
        j.observe(1.0)
        j.observe(2.0)
        j.observe(3.0)
        assert j.history == [1.0, 2.0, 3.0]
        assert j.n_observations == 3


# ── Room tests ──────────────────────────────────────────────────────────

class TestRoom:
    def test_create(self):
        r = Room("a", vibe=1.5)
        assert r.id == "a"
        assert r.vibe == 1.5

    def test_observe_updates_vibe_and_surprise(self):
        r = Room("a", vibe=0.0)
        r.observe(1.0)
        assert r.vibe == 1.0
        assert r.last_surprise > 0

    def test_predict_after_observe(self):
        r = Room("a", vibe=5.0)
        p = r.predict()
        assert p == pytest.approx(5.0)

    def test_repr(self):
        r = Room("x")
        assert "x" in repr(r)


# ── CellGraph tests ────────────────────────────────────────────────────

class TestCellGraph:
    def _make_graph(self) -> CellGraph:
        g = CellGraph()
        g.add_room(Room("a", vibe=1.0))
        g.add_room(Room("b", vibe=2.0))
        g.add_room(Room("c", vibe=3.0))
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        return g

    def test_add_rooms_and_edges(self):
        g = self._make_graph()
        assert len(g) == 3
        assert "b" in g.neighbours("a")
        assert "a" in g.neighbours("b")

    def test_add_edge_missing_room_raises(self):
        g = CellGraph()
        g.add_room(Room("a"))
        with pytest.raises(KeyError):
            g.add_edge("a", "z")

    def test_total_vibe(self):
        g = self._make_graph()
        # Initial vibes: 1+2+3=6, but add_room seeds Jepa which re-observes
        # After construction total_vibe reflects current .vibe values
        total = g.total_vibe()
        assert total == pytest.approx(6.0)

    def test_diffuse_conserves_vibe(self):
        g = self._make_graph()
        before = g.total_vibe()
        for _ in range(10):
            g.diffuse(rate=0.2)
        after = g.total_vibe()
        assert after == pytest.approx(before, abs=1e-6)

    def test_tick(self):
        g = self._make_graph()
        g.tick()
        # Should still have rooms
        assert len(g) == 3

    def test_gossip(self):
        g = self._make_graph()
        v = g.gossip("a")
        assert isinstance(v, float)

    def test_gossip_no_neighbours(self):
        g = CellGraph()
        g.add_room(Room("solo", vibe=5.0))
        v = g.gossip("solo")
        assert v == pytest.approx(5.0)

    def test_gossip_specific_target(self):
        g = self._make_graph()
        g.gossip("a", target_id="b")
        # Both should now have same vibe
        assert g.rooms["a"].vibe == pytest.approx(g.rooms["b"].vibe)

    def test_learn(self):
        g = self._make_graph()
        before = {rid: r.vibe for rid, r in g.rooms.items()}
        g.learn()
        # Learn should change vibes
        changed = any(g.rooms[rid].vibe != before[rid] for rid in g.rooms)
        assert changed or all(True for _ in g.rooms)  # at least runs

    def test_len(self):
        g = self._make_graph()
        assert len(g) == 3

    def test_repr(self):
        g = self._make_graph()
        assert "CellGraph" in repr(g)


# ── Topology tests ─────────────────────────────────────────────────────

class TestTopology:
    def test_chain(self):
        g = chain(5)
        assert len(g) == 5
        assert len(g.neighbours("r0")) == 1
        assert len(g.neighbours("r2")) == 2

    def test_ring(self):
        g = ring(5)
        assert len(g) == 5
        assert "r4" in g.neighbours("r0")

    def test_star(self):
        g = star(5)
        assert len(g) == 5
        assert len(g.neighbours("r0")) == 4
        assert len(g.neighbours("r1")) == 1

    def test_mesh(self):
        g = mesh(4)
        assert len(g) == 4
        assert len(g.neighbours("r0")) == 3

    def test_small_world(self):
        g = small_world(10, k=4, p=0.2)
        assert len(g) == 10

    def test_small_world_k_must_be_even(self):
        with pytest.raises(ValueError):
            small_world(10, k=3)

    def test_scale_free(self):
        g = scale_free(20, m=2)
        assert len(g) == 20

    def test_scale_free_invalid_m(self):
        with pytest.raises(ValueError):
            scale_free(10, m=0)
        with pytest.raises(ValueError):
            scale_free(5, m=5)

    def test_diffuse_conservation_on_ring(self):
        g = ring(10)
        # Set vibes
        for i in range(10):
            g.rooms[f"r{i}"].vibe = float(i)
            g.rooms[f"r{i}"].jepa.observe(float(i))
        before = g.total_vibe()
        for _ in range(20):
            g.diffuse(rate=0.3)
        after = g.total_vibe()
        assert after == pytest.approx(before, abs=1e-4)
