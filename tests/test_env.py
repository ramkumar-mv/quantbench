from quantbench.env import capture_env


def test_capture_env_has_required_keys():
    env = capture_env()
    assert {"python", "platform", "machine", "packages", "gpu"} <= set(env)
    assert isinstance(env["packages"], dict)
