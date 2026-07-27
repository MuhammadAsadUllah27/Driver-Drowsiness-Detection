import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import infer_cfg
from inference.state_machine import DrowsinessStateMachine, FrameFeatures


def test_state_machine_flags_drowsy_for_cnn_and_eye_cues():
    state_machine = DrowsinessStateMachine()
    feat = FrameFeatures(
        cnn_prob=0.70,
        ear=0.20,
        mar=0.0,
        pitch=0.0,
        yaw=0.0,
        face_detected=True,
    )

    state = None
    for _ in range(infer_cfg.alert_trigger_frames):
        state = state_machine.update(feat)

    assert state is not None
    assert state.label == "DROWSY"
