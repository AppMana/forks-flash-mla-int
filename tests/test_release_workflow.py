from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_container_wheels_use_a_restorable_local_sccache() -> None:
    workflow = (ROOT / ".github" / "workflows" / "wheels.yaml").read_text()

    assert workflow.count("uses: actions/cache@v4") >= 2
    assert workflow.count('-e SCCACHE_DIR=/sccache-cache') >= 2
    assert workflow.count('${RUNNER_TEMP}/sccache:/sccache-cache') >= 2
