import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cloud_contracts import sanitize_headers, sanitize_profile_payload, sanitize_command


def test_sanitize_headers_removes_sensitive_values():
    headers = sanitize_headers({
        "Authorization": "Bearer secret",
        "Cookie": "session=secret",
        "Content-Type": "application/json",
        "X-Request-Id": "abc",
    })

    assert "authorization" not in headers
    assert "cookie" not in headers
    assert headers["content-type"] == "application/json"
    assert headers["x-request-id"] == "abc"


def test_sanitize_profile_payload_normalizes_shape_and_drops_raw_embedding():
    profile = sanitize_profile_payload({
        "profile": {
            "tinder_id": "tid123",
            "name": " Ana  Maria ",
            "age": "23",
            "interests": "musica, viagem, musica",
            "photo_features": {
                "photo_has_face": 1,
                "_embedding": [1, 2, 3],
                "_faces_found": 1,
            },
        }
    })

    assert profile["_tinder_id"] == "tid123"
    assert profile["name"] == "Ana Maria"
    assert profile["age"] == 23
    assert profile["interests"] == ["musica", "viagem"]
    assert profile["_photo_features"]["photo_has_face"] == 1.0
    assert profile["_photo_features"]["_faces_found"] == 1.0
    assert "_embedding" not in profile["_photo_features"]


def test_sanitize_command_rejects_unknown_commands():
    try:
        sanitize_command({"command": "rm_everything"})
    except ValueError as exc:
        assert "comando nao permitido" in str(exc)
    else:
        raise AssertionError("unknown command should be rejected")
