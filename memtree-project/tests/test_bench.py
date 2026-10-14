from memtree.bench import make_session, run_session, scaling_table


def test_session_recall_is_perfect_and_reads_are_logarithmic():
    for seed in range(3):
        r = run_session(make_session(seed=seed, n_events=1500))
        writes = sum(r["status"].values())
        assert r["recall"] == 1.0
        assert r["log_bound_violations"] == 0
        assert writes + r["queries"] == r["events"]
        assert r["status"].get("refresh", 0) > 0 and r["status"].get("update", 0) > 0
        assert r["bridges"] > 0  # shared facts bridge across peers
        assert r["max_read_steps"] <= r["max_depths"]
        assert r["reads_by_zone"].get("watchdog", 0) > 0


def test_session_is_deterministic():
    a = run_session(make_session(seed=5, n_events=800))
    b = run_session(make_session(seed=5, n_events=800))
    assert a == b


def test_scaling_dense_quadratic_sparse_linear_tree_nlogn():
    rows = scaling_table(sizes=(256, 512, 1024, 2048))
    for a, b in zip(rows, rows[1:]):
        assert 3.8 < b["dense_keys"] / a["dense_keys"] < 4.2
        assert b["sparse_keys"] / a["sparse_keys"] < 2.1
        assert 1.9 < b["tree_states"] / a["tree_states"] < 2.4
    assert all(r["max_states"] <= r["log2T_plus_1"] for r in rows)
