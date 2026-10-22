from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("configured", ["example-group", "group:example-group", "*"])
def test_signal_group_allowlist_accepts_prefixed_inbound_id(configured, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.pairing_store = SimpleNamespace(is_approved=lambda *args, **kwargs: False)
    source = SessionSource(
        platform=Platform.SIGNAL, chat_id="group:example-group", chat_type="group",
        user_id=None, user_name="",
    )
    assert runner._is_user_authorized(source) is False
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", configured)
    assert runner._is_user_authorized(source) is True
