"""
Tests — Decentralized Architecture
=====================================
Unit tests for:
  1. BaseNode registration and webhook dispatch
  2. KME session store (create, join, qubit bus, measurement bus)
  3. Sifting and key lifecycle
  4. Node event handler wiring
  5. Config-driven topology (network.yaml parsing)
"""

import sys, os, json, time, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock, AsyncMock, patch
from models import (
    NodeRole, NodeRegistration, NodeInfo,
    SessionCreateReq, SessionJoinReq,
    QubitRecord, QubitBatch, QubitUpload,
    MeasurementRecord, MeasurementUpload,
    SiftUpload, KeyUpload, KeyStatus,
    WebhookEvent, Basis,
)


# ─────────────────────────────────────────────
# Mock Redis
# ─────────────────────────────────────────────

def make_mock_redis() -> tuple[MagicMock, dict, dict]:
    store = {}
    sets  = {}
    lists = {}
    hashes = {}

    r = MagicMock()
    r.set.side_effect     = lambda k, v, ex=None: store.update({k: v})
    r.get.side_effect     = lambda k: store.get(k)
    r.delete.side_effect  = lambda *keys: [store.pop(k, None) for k in keys]
    r.sadd.side_effect    = lambda key, *vals: sets.setdefault(key, set()).update(vals)
    r.srem.side_effect    = lambda key, *vals: [sets.get(key, set()).discard(v) for v in vals]
    r.smembers.side_effect= lambda key: sets.get(key, set())
    r.rpush.side_effect   = lambda key, val: lists.setdefault(key, []).append(val)
    r.lpop.side_effect    = lambda key: lists.get(key, [None]).pop(0) if lists.get(key) else None
    r.llen.side_effect    = lambda key: len(lists.get(key, []))
    r.hset.side_effect    = lambda name, key=None, value=None, mapping=None: (
        hashes.setdefault(name, {}).update(
            {key: value} if key else (mapping or {})
        )
    )
    r.hgetall.side_effect = lambda name: hashes.get(name, {})
    r.expire.return_value = True
    r.ping.return_value   = True

    return r, store, sets


# ─────────────────────────────────────────────
# 1. Node registry tests
# ─────────────────────────────────────────────

def test_register_node_assigns_id():
    from kme.node_registry import register_node, load_node
    r, store, _ = make_mock_redis()

    reg  = NodeRegistration(
        role=NodeRole.SENDER,
        callback_url="http://localhost:8001/webhook",
        label="alice-test",
    )
    info = register_node(r, reg)

    assert info.node_id is not None
    assert info.role    == NodeRole.SENDER
    assert info.label   == "alice-test"

    loaded = load_node(r, info.node_id)
    assert loaded.node_id == info.node_id
    print("✓ test_register_node_assigns_id")


def test_find_node_by_label():
    from kme.node_registry import register_node, find_node_by_label
    r, _, _ = make_mock_redis()

    for label in ["alice-1", "bob-1", "alice-2"]:
        role = NodeRole.SENDER if "alice" in label else NodeRole.RECEIVER
        register_node(r, NodeRegistration(
            role=role,
            callback_url=f"http://localhost/webhook",
            label=label,
        ))

    found = find_node_by_label(r, "bob-1")
    assert found is not None
    assert found.role == NodeRole.RECEIVER

    not_found = find_node_by_label(r, "eve-99")
    assert not_found is None
    print("✓ test_find_node_by_label")


def test_list_nodes_by_role():
    from kme.node_registry import register_node, list_nodes
    r, _, _ = make_mock_redis()

    register_node(r, NodeRegistration(role=NodeRole.SENDER,   callback_url="http://a", label="a1"))
    register_node(r, NodeRegistration(role=NodeRole.SENDER,   callback_url="http://a", label="a2"))
    register_node(r, NodeRegistration(role=NodeRole.RECEIVER, callback_url="http://b", label="b1"))

    senders   = list_nodes(r, role=NodeRole.SENDER)
    receivers = list_nodes(r, role=NodeRole.RECEIVER)
    all_nodes = list_nodes(r)

    assert len(senders)   == 2
    assert len(receivers) == 1
    assert len(all_nodes) == 3
    print("✓ test_list_nodes_by_role")


# ─────────────────────────────────────────────
# 2. Session store tests
# ─────────────────────────────────────────────

def test_save_and_load_session():
    from kme.session_store import save_session, load_session
    r, _, _ = make_mock_redis()

    session = {
        "session_id":      "sess-001",
        "status":          "open",
        "sender_node_id":  "node-alice",
        "receiver_node_id": "node-bob",
        "n_qubits":        200,
        "batch_size":      10,
        "loss_rate":       0.0,
        "retry_enabled":   False,
        "created_at":      time.time(),
        "key_status":      KeyStatus.NONE.value,
        "key_expires_at":  None,
        "n_delivered":     0,
        "n_sifted":        0,
        "qber":            0.0,
        "key_final":       "",
        "error_message":   "",
    }
    save_session(r, session)
    loaded = load_session(r, "sess-001")

    assert loaded["session_id"]      == "sess-001"
    assert loaded["n_qubits"]        == 200
    assert loaded["sender_node_id"]  == "node-alice"
    print("✓ test_save_and_load_session")


def test_update_session_patches_fields():
    from kme.session_store import save_session, load_session, update_session
    r, _, _ = make_mock_redis()

    session = {"session_id": "sess-002", "status": "open",
               "sender_node_id": "a", "receiver_node_id": "b",
               "n_qubits": 100, "batch_size": 10, "loss_rate": 0.0,
               "retry_enabled": False, "created_at": time.time(),
               "key_status": "none", "key_expires_at": None,
               "n_delivered": 0, "n_sifted": 0, "qber": 0.0,
               "key_final": "", "error_message": ""}
    save_session(r, session)
    update_session(r, "sess-002", status="joined", n_delivered=150)

    loaded = load_session(r, "sess-002")
    assert loaded["status"]      == "joined"
    assert loaded["n_delivered"] == 150
    assert loaded["n_qubits"]    == 100   # untouched
    print("✓ test_update_session_patches_fields")


def test_qubit_bus_push_pop():
    from kme.session_store import push_qubit_batch, pop_qubit_batch, qubit_batch_count
    r, _, _ = make_mock_redis()

    batch0 = {"batch_id": 0, "session_id": "s", "qubits": []}
    batch1 = {"batch_id": 1, "session_id": "s", "qubits": []}

    push_qubit_batch(r, "s", batch0)
    push_qubit_batch(r, "s", batch1)

    assert qubit_batch_count(r, "s") == 2

    popped = pop_qubit_batch(r, "s")
    assert popped["batch_id"] == 0    # FIFO

    assert qubit_batch_count(r, "s") == 1
    print("✓ test_qubit_bus_push_pop")


def test_measurement_bus_save_load():
    from kme.session_store import save_measurements, load_measurements
    r, _, _ = make_mock_redis()

    upload = {
        "session_id":   "sess-meas",
        "node_id":      "bob-1",
        "measurements": [
            {"qubit_id": 0, "basis": "Z", "bit_result": 1},
            {"qubit_id": 1, "basis": "X", "bit_result": 0},
            {"qubit_id": 2, "basis": "Z", "bit_result": 1},
        ]
    }
    save_measurements(r, "sess-meas", upload)
    loaded = load_measurements(r, "sess-meas")

    assert len(loaded)         == 3
    assert loaded[0]["basis"]  == "Z"
    assert loaded[1]["basis"]  == "X"
    assert loaded[2]["bit_result"] == 1
    print("✓ test_measurement_bus_save_load")


def test_sift_bus_save_load():
    from kme.session_store import save_sift_upload, load_sift_upload
    r, _, _ = make_mock_redis()

    upload = {
        "session_id":  "sess-sift",
        "alice_bases": [(0, "Z"), (1, "X"), (2, "Z")],
        "sample_seed": 42,
    }
    save_sift_upload(r, "sess-sift", upload)
    loaded = load_sift_upload(r, "sess-sift")

    assert loaded["sample_seed"]   == 42
    assert len(loaded["alice_bases"]) == 3
    print("✓ test_sift_bus_save_load")


# ─────────────────────────────────────────────
# 3. Key lifecycle tests
# ─────────────────────────────────────────────

def _base_session(session_id: str) -> dict:
    return {
        "session_id": session_id, "status": "done",
        "sender_node_id": "a", "receiver_node_id": "b",
        "n_qubits": 100, "batch_size": 10, "loss_rate": 0.0,
        "retry_enabled": False, "created_at": time.time(),
        "key_status": KeyStatus.NONE.value, "key_expires_at": None,
        "n_delivered": 95, "n_sifted": 45, "qber": 0.02,
        "key_final": "101010110011", "error_message": "",
    }


def test_key_activate_and_consume():
    from kme.session_store import (
        save_session, load_session, activate_key, consume_key
    )
    r, _, _ = make_mock_redis()

    session = _base_session("key-sess-1")
    save_session(r, session)

    expires_at = activate_key(r, "key-sess-1")
    assert expires_at > time.time()

    loaded = load_session(r, "key-sess-1")
    assert loaded["key_status"] == KeyStatus.ACTIVE.value

    ok, key = consume_key(r, "key-sess-1")
    assert ok  == True
    assert key == "101010110011"

    loaded2 = load_session(r, "key-sess-1")
    assert loaded2["key_status"] == KeyStatus.CONSUMED.value
    print("✓ test_key_activate_and_consume")


def test_key_double_consume_fails():
    from kme.session_store import save_session, activate_key, consume_key
    r, _, _ = make_mock_redis()

    session = _base_session("key-sess-2")
    save_session(r, session)
    activate_key(r, "key-sess-2")

    ok1, _ = consume_key(r, "key-sess-2")
    ok2, _ = consume_key(r, "key-sess-2")   # second attempt

    assert ok1 == True
    assert ok2 == False   # already consumed
    print("✓ test_key_double_consume_fails")


def test_key_expired_cannot_consume():
    from kme.session_store import save_session, update_session, consume_key
    r, _, _ = make_mock_redis()

    session = _base_session("key-sess-3")
    save_session(r, session)

    # Manually set expired state (TTL already elapsed)
    update_session(r, "key-sess-3",
                   key_status=KeyStatus.ACTIVE.value,
                   key_expires_at=time.time() - 1.0)   # expired 1s ago

    ok, key = consume_key(r, "key-sess-3")
    assert ok  == False
    assert key is None
    print("✓ test_key_expired_cannot_consume")


def test_key_none_cannot_consume():
    """Cannot consume a key that was never activated."""
    from kme.session_store import save_session, consume_key
    r, _, _ = make_mock_redis()

    session = _base_session("key-sess-4")
    save_session(r, session)   # key_status=NONE

    ok, key = consume_key(r, "key-sess-4")
    assert ok  == False
    assert key is None
    print("✓ test_key_none_cannot_consume")


# ─────────────────────────────────────────────
# 4. Webhook event dispatch tests
# ─────────────────────────────────────────────

def test_webhook_dispatches_to_correct_handler():
    """
    WebhookEvent with event='session_open' must call on_session_open.
    We verify the dispatch table in BaseNode.handle_webhook.
    """
    from node.base_node import BaseNode

    # Minimal concrete subclass for testing
    class TestNode(BaseNode):
        def __init__(self):
            super().__init__(
                role=NodeRole.RECEIVER,
                label="test-node",
                callback_url="http://localhost/webhook",
            )
            self.received_events: list[str] = []

        async def on_session_open(self, sid, payload):
            self.received_events.append(f"session_open:{sid}")

        async def on_key_available(self, sid, payload):
            self.received_events.append(f"key_available:{sid}")

    node = TestNode()
    node.node_id = "fake-id"

    async def run():
        await node.handle_webhook(WebhookEvent(
            event="session_open",
            session_id="sess-abc",
            payload={"role": "receiver"},
        ))
        await asyncio.sleep(0.05)   # let the task run

        await node.handle_webhook(WebhookEvent(
            event="key_available",
            session_id="sess-abc",
            payload={"qber": 0.02},
        ))
        await asyncio.sleep(0.05)

    asyncio.run(run())

    assert "session_open:sess-abc"  in node.received_events
    assert "key_available:sess-abc" in node.received_events
    print("✓ test_webhook_dispatches_to_correct_handler")


def test_unknown_webhook_event_does_not_crash():
    from node.base_node import BaseNode

    class TestNode(BaseNode):
        def __init__(self):
            super().__init__(NodeRole.SENDER, "t", "http://x/webhook")

    node = TestNode()
    node.node_id = "fake-id"

    async def run():
        # Should not raise
        await node.handle_webhook(WebhookEvent(
            event="unknown_future_event",
            session_id="sess-xyz",
            payload={},
        ))

    asyncio.run(run())
    print("✓ test_unknown_webhook_event_does_not_crash")


# ─────────────────────────────────────────────
# 5. BB84 sifting logic (unchanged from v6)
# ─────────────────────────────────────────────

def test_sifting_by_qubit_id_order_invariant():
    """
    Core BB84 invariant: sifting result must be identical
    regardless of the order measurements arrive in.
    This is preserved in the new architecture.
    """
    from bb84_logic import perform_sifting_by_id
    from models import MeasurementRecord as QubitMeasurement

    alice_bits  = [1, 0, 1, 0, 1]
    # Use the actual Basis enum values ("Z" and "X") to match models.py
    alice_bases = ["Z", "X", "Z", "X", "Z"]

    # Measurements arrive out of order (simulates network reordering)
    meas_map = {
        4: QubitMeasurement(qubit_id=4, basis=Basis.RECTILINEAR, bit_result=1),
        0: QubitMeasurement(qubit_id=0, basis=Basis.RECTILINEAR, bit_result=1),
        2: QubitMeasurement(qubit_id=2, basis=Basis.RECTILINEAR, bit_result=1),
        1: QubitMeasurement(qubit_id=1, basis=Basis.DIAGONAL,    bit_result=0),
        3: QubitMeasurement(qubit_id=3, basis=Basis.DIAGONAL,    bit_result=0),
    }

    a, b, ids = perform_sifting_by_id(alice_bits, alice_bases, meas_map)

    # All 5 bases match → all retained, sorted by qubit_id
    assert ids == [0, 1, 2, 3, 4]
    assert a   == [1, 0, 1, 0, 1]
    print("✓ test_sifting_by_qubit_id_order_invariant")


def test_qber_zero_with_no_noise():
    from bb84_logic import compute_qber
    key = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1] * 4   # 48 bits
    qber, alice_k, bob_k = compute_qber(key, key[:], sample_seed=99)
    assert qber == 0.0
    assert alice_k == bob_k
    print(f"✓ test_qber_zero_with_no_noise (key_len={len(alice_k)})")


# ─────────────────────────────────────────────
# 6. network.yaml parsing
# ─────────────────────────────────────────────

def test_network_yaml_loads():
    """network.yaml must be parseable and contain required fields."""
    import yaml
    config_path = os.path.join(
        os.path.dirname(__file__), 'nodes', 'network.yaml'
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    assert "kme"   in cfg
    assert "qkdl"  in cfg
    assert "nodes" in cfg
    assert len(cfg["nodes"]) >= 2

    for node in cfg["nodes"]:
        assert "label"  in node
        assert "role"   in node
        assert "module" in node
        assert "port"   in node

    print(f"✓ test_network_yaml_loads ({len(cfg['nodes'])} nodes configured)")


def test_network_yaml_roles_valid():
    """All node roles must be valid NodeRole values."""
    import yaml
    config_path = os.path.join(
        os.path.dirname(__file__), 'nodes', 'network.yaml'
    )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    valid_roles = {r.value for r in NodeRole}
    for node in cfg["nodes"]:
        assert node["role"] in valid_roles, \
            f"Invalid role '{node['role']}' for node '{node['label']}'"

    print("✓ test_network_yaml_roles_valid")


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 58)
    print("  Tests — Decentralized Agent Architecture")
    print("═" * 58)

    print("\n── Node registry ───────────────────────────────────")
    test_register_node_assigns_id()
    test_find_node_by_label()
    test_list_nodes_by_role()

    print("\n── Session store ───────────────────────────────────")
    test_save_and_load_session()
    test_update_session_patches_fields()
    test_qubit_bus_push_pop()
    test_measurement_bus_save_load()
    test_sift_bus_save_load()

    print("\n── Key lifecycle ───────────────────────────────────")
    test_key_activate_and_consume()
    test_key_double_consume_fails()
    test_key_expired_cannot_consume()
    test_key_none_cannot_consume()

    print("\n── Webhook dispatch ────────────────────────────────")
    test_webhook_dispatches_to_correct_handler()
    test_unknown_webhook_event_does_not_crash()

    print("\n── BB84 logic (unchanged) ──────────────────────────")
    test_sifting_by_qubit_id_order_invariant()
    test_qber_zero_with_no_noise()

    print("\n── Network config ──────────────────────────────────")
    test_network_yaml_loads()
    test_network_yaml_roles_valid()

    print("\n✓ All tests pass.")
    print("═" * 58)
