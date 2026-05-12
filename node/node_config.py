"""
node/node_config.py
====================
Loads node identity and runtime settings from environment variables
or from network.yaml.

Priority order:
  1. Environment variables (set by node_runner.py from network.yaml)
  2. network.yaml defaults
  3. Hardcoded fallbacks below

Why a config module instead of reading os.environ directly in each node?
  Single point of truth for all node settings.
  Easy to mock in tests without patching os.environ everywhere.
  Validated types (int ports, float intervals, bool flags).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "nodes" / "network.yaml"


@dataclass
class NodeConfig:
    """
    Runtime configuration for a single node.
    Passed to BaseNode.__init__() at startup.
    """
    # Identity
    label:        str
    role:         str        # "sender" | "receiver" | "relay" | "monitor"
    port:         int

    # URLs
    kme_url:      str
    qkdl_url:     str
    my_url:       str        # this node's own base URL (for webhook callback)

    # Behaviour
    batch_size:   int        = 10
    poll_interval: float     = 2.0
    key_ttl:      int        = 300

    # Optional
    redis_url:    str        = "redis://localhost:6379/0"
    log_level:    str        = "INFO"
    metadata:     dict       = None   # arbitrary node metadata

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def callback_url(self) -> str:
        """The URL the KME uses to send webhooks to this node."""
        return f"{self.my_url}/webhook"


def load_from_env(label: str, port: int) -> NodeConfig:
    """
    Builds a NodeConfig from environment variables.
    This is what node_runner.py uses — it sets env vars from network.yaml
    before spawning each subprocess.

    Environment variables (all optional with fallbacks):
      KME_URL, QKDL_URL, REDIS_URL
      ALICE_LABEL / BOB_LABEL / NODE_LABEL  → node label
      ALICE_URL / BOB_URL / NODE_URL         → this node's base URL
      BATCH_SIZE, NODE_POLL_INTERVAL, BB84_KEY_TTL
    """
    role   = os.getenv("NODE_ROLE", _infer_role(label))
    my_url = os.getenv("NODE_URL") \
          or os.getenv("ALICE_URL") \
          or os.getenv("BOB_URL") \
          or f"http://localhost:{port}"

    return NodeConfig(
        label         = label,
        role          = role,
        port          = port,
        kme_url       = os.getenv("KME_URL",            "http://localhost:8000"),
        qkdl_url      = os.getenv("QKDL_URL",           "http://localhost:8003"),
        my_url        = my_url,
        batch_size    = int(os.getenv("BATCH_SIZE",      "10")),
        poll_interval = float(os.getenv("NODE_POLL_INTERVAL", "2.0")),
        key_ttl       = int(os.getenv("BB84_KEY_TTL",   "300")),
        redis_url     = os.getenv("REDIS_URL",          "redis://localhost:6379/0"),
        log_level     = os.getenv("LOG_LEVEL",          "INFO"),
    )


def load_from_yaml(label: str) -> Optional[NodeConfig]:
    """
    Loads config for a specific node label from network.yaml.
    Used when running a node directly (not via node_runner.py).
    Returns None if the label is not found.
    """
    if not _CONFIG_PATH.exists():
        return None

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    kme_url  = cfg.get("kme",  {}).get("url",  "http://localhost:8000")
    qkdl_url = cfg.get("qkdl", {}).get("url",  "http://localhost:8003")

    for node in cfg.get("nodes", []):
        if node["label"] == label:
            env     = node.get("env", {})
            port    = node["port"]
            my_url  = env.get("ALICE_URL") or env.get("BOB_URL") \
                   or f"http://localhost:{port}"
            return NodeConfig(
                label         = label,
                role          = node["role"],
                port          = port,
                kme_url       = kme_url,
                qkdl_url      = qkdl_url,
                my_url        = my_url,
                batch_size    = int(env.get("BATCH_SIZE", "10")),
                poll_interval = 2.0,
                key_ttl       = cfg.get("kme", {}).get("key_ttl_seconds", 300),
                redis_url     = "redis://localhost:6379/0",
            )
    return None


def load(label: str, port: int) -> NodeConfig:
    """
    Loads config with priority: env → yaml → fallback.
    This is the single function nodes should call.

    Example (in alice/main.py):
        from node.node_config import load as load_config
        config = load_config(label="alice-1", port=8001)
        alice  = AliceNode(config)
    """
    # If KME_URL is set in environment, env takes priority
    if os.getenv("KME_URL"):
        return load_from_env(label, port)

    # Try network.yaml
    cfg = load_from_yaml(label)
    if cfg:
        return cfg

    # Fallback
    return load_from_env(label, port)


def _infer_role(label: str) -> str:
    """Infers role from label convention (alice-* → sender, bob-* → receiver)."""
    label_lower = label.lower()
    if "alice" in label_lower:
        return "sender"
    if "bob"   in label_lower:
        return "receiver"
    if "relay" in label_lower:
        return "relay"
    return "monitor"


# ─────────────────────────────────────────────
# Quick check
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load("alice-1", 8001)
    print(f"label        : {cfg.label}")
    print(f"role         : {cfg.role}")
    print(f"kme_url      : {cfg.kme_url}")
    print(f"callback_url : {cfg.callback_url}")
    print(f"batch_size   : {cfg.batch_size}")
