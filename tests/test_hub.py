from lerobot_mcp.hub import _extract_robots


def test_extract_robots_matches_so101_aliases() -> None:
    values = [
        "JYeonKim/hil_lerobot_SO-101_wettissue_move_set_smolvla_ep100_re",
        "tuuy/lerobot_so_arm101_task0_new",
    ]

    assert _extract_robots(values) == ["so101"]
