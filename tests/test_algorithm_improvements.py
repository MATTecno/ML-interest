import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dataset
import photo_deep_feedback
import photo_features
import photo_storage
import review_queue
import review_ui
import state
import model_training
import photo_semantic_embeddings
from decision_policy import apply_decision_policy
from features import (
    BODY_PREFERENCE_FEATURE_NAMES,
    GENDER_PROXY_FEATURE_NAMES,
    MODEL_PHOTO_FEATURE_NAMES,
    RACE_FEATURE_NAMES,
    SEMANTIC_EMBEDDING_FEATURE_NAMES,
    extract_features,
)
from model_evaluation import optimize_like_threshold
from predictor import _apply_race_affinity, _gender_compatibility_filter
from profile_parser import extract_descriptors, parse_profile
from resource_guard import is_memory_pressure, memory_pressure_photo_features
from review_ui import _body_measurement_strength


class AlgorithmImprovementTests(unittest.TestCase):
    def test_decision_policy_uses_recall_safe_threshold_and_review_band(self):
        result = apply_decision_policy(
            0.49,
            config={"model": {"decision_policy": {"fallback_like_threshold": 0.48, "review_band": [0.4, 0.6]}}},
            model_data={},
        )

        self.assertEqual(result["decision"], "CURTIR")
        self.assertTrue(result["in_review_band"])
        self.assertGreaterEqual(result["review_priority"], 0.85)
        self.assertEqual(result["preference_tier"], "like")

    def test_hard_filter_policy_forces_filtered_pass(self):
        result = apply_decision_policy(0.99, forced_pass=True, filter_reason="nome bloqueado")

        self.assertEqual(result["decision"], "NÃO CURTIR")
        self.assertEqual(result["preference_tier"], "filtered_pass")
        self.assertEqual(result["filter_reason"], "nome bloqueado")

    def test_body_and_race_are_model_features_but_gender_is_filter_only(self):
        self.assertTrue(any(name in MODEL_PHOTO_FEATURE_NAMES for name in BODY_PREFERENCE_FEATURE_NAMES))
        self.assertTrue(any(name in MODEL_PHOTO_FEATURE_NAMES for name in RACE_FEATURE_NAMES))
        self.assertTrue(all(name not in MODEL_PHOTO_FEATURE_NAMES for name in GENDER_PROXY_FEATURE_NAMES))

    def test_race_affinity_adjusts_probability_and_gender_filter_forces_pass(self):
        adjusted, details = _apply_race_affinity(
            0.50,
            {"photo_race_black": 0.8},
            {"preferences": {"race_affinity": {"black": 0.10}}},
        )
        self.assertGreater(adjusted, 0.50)
        self.assertTrue(details["applied"])

        gender = _gender_compatibility_filter(
            {
                "photo_has_face": 1.0,
                "photo_woman_confidence": 0.05,
                "photo_gender_certainty": 0.90,
            },
            {
                "model": {
                    "gender_compatibility": {
                        "enabled": True,
                        "target": "woman",
                        "min_woman_confidence": 0.20,
                        "min_certainty_for_filter": 0.65,
                    }
                }
            },
        )
        self.assertTrue(gender["forced_pass"])

    def test_threshold_optimizer_prefers_recall_with_precision_floor(self):
        y = [1, 1, 1, 1, 0, 0, 0, 0]
        p = [0.91, 0.74, 0.52, 0.49, 0.60, 0.42, 0.30, 0.20]

        result = optimize_like_threshold(y, p, precision_floor=0.55, target_like_recall=0.75)

        self.assertGreaterEqual(result["like_threshold"], 0.45)
        self.assertLessEqual(result["like_threshold"], 0.55)

    def test_dataset_schema_migrates_new_label_columns_and_excludes_maybe_from_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_profiles = Path(tmp) / "profiles.csv"
            synthetic = Path(tmp) / "synthetic.csv"
            old_profiles.write_text(
                "name,age,label,source,final_decision,feedback_details\n"
                "A,20,1,real,CURTIR,\n"
                "B,21,0,real,NÃO CURTIR,\n"
                "C,22,,real,TALVEZ,\n",
                encoding="utf-8",
            )
            synthetic.write_text("name,age,label,source\n", encoding="utf-8")

            old_profiles_path = dataset.PROFILES_PATH
            old_synthetic_path = dataset.SYNTHETIC_PATH
            try:
                dataset.PROFILES_PATH = old_profiles
                dataset.SYNTHETIC_PATH = synthetic
                dataset._ensure_profiles_schema()
                with old_profiles.open(encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))

                self.assertIn("preference_tier", rows[0])
                self.assertIn("correction_type", rows[0])
                self.assertIn("photo_clip_pc_01", rows[0])
                self.assertIn("photo_semantic_embedding_saved", rows[0])
                self.assertEqual(dataset.count_real_profiles(), 3)
                self.assertEqual(dataset.count_trainable_real_profiles(), 2)
                self.assertEqual(len(dataset.load_all_data()), 2)
            finally:
                dataset.PROFILES_PATH = old_profiles_path
                dataset.SYNTHETIC_PATH = old_synthetic_path

    def test_review_queue_skips_profile_already_trained_by_name_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = Path(tmp) / "profiles.csv"
            review_path = Path(tmp) / "review_queue.csv"
            profiles.write_text(
                "name,age,label,source,bio,interests\n"
                "Julia,22,1,real,bio antiga,academia\n",
                encoding="utf-8",
            )

            old_review_path = review_queue.REVIEW_PATH
            old_profiles_path = review_queue.PROFILES_PATH
            try:
                review_queue.REVIEW_PATH = review_path
                review_queue.PROFILES_PATH = profiles
                review_id = review_queue.enqueue_auto_decision(
                    {
                        "name": "Julia",
                        "age": "22",
                        "bio": "bio nova",
                        "interests": ["viagem"],
                        "_photo_features": {},
                    },
                    "CURTIR",
                    "CURTIR",
                )
                self.assertEqual(review_id, "")
                self.assertFalse(review_path.exists())
            finally:
                review_queue.REVIEW_PATH = old_review_path
                review_queue.PROFILES_PATH = old_profiles_path

    def test_review_queue_dedupes_signed_photo_urls_by_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = Path(tmp) / "profiles.csv"
            review_path = Path(tmp) / "review_queue.csv"
            profiles.write_text("name,age,label,source\n", encoding="utf-8")

            old_review_path = review_queue.REVIEW_PATH
            old_profiles_path = review_queue.PROFILES_PATH
            try:
                review_queue.REVIEW_PATH = review_path
                review_queue.PROFILES_PATH = profiles
                first_id = review_queue.enqueue_auto_decision(
                    {
                        "name": "Julia",
                        "age": "22",
                        "bio": "",
                        "interests": [],
                        "_photo_url": "https://images-ssl.gotinder.com/u/userA/photo1.webp?Policy=aaa",
                        "_photo_features": {},
                    },
                    "CURTIR",
                    "CURTIR",
                )
                second_id = review_queue.enqueue_auto_decision(
                    {
                        "name": "Outro nome",
                        "age": "22",
                        "bio": "",
                        "interests": [],
                        "_photo_url": "https://images-ssl.gotinder.com/u/userA/photo1.webp?Policy=bbb",
                        "_photo_features": {},
                    },
                    "CURTIR",
                    "CURTIR",
                )

                self.assertEqual(second_id, first_id)
                with review_path.open(encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 1)
                self.assertIn("path:images-ssl.gotinder.com/u/userA/photo1.webp", rows[0]["photo_url_keys"])
            finally:
                review_queue.REVIEW_PATH = old_review_path
                review_queue.PROFILES_PATH = old_profiles_path

    def test_review_queue_skips_trained_profile_by_photo_owner_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = Path(tmp) / "profiles.csv"
            review_path = Path(tmp) / "review_queue.csv"

            with profiles.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=dataset.CSV_FIELDNAMES)
                writer.writeheader()
                row = {key: "" for key in dataset.CSV_FIELDNAMES}
                row.update({"name": "Antiga", "age": "22", "source": "real", "label": "1", "photo_url_key": "owner:userA"})
                writer.writerow(row)

            old_review_path = review_queue.REVIEW_PATH
            old_profiles_path = review_queue.PROFILES_PATH
            try:
                review_queue.REVIEW_PATH = review_path
                review_queue.PROFILES_PATH = profiles
                review_id = review_queue.enqueue_auto_decision(
                    {
                        "name": "Julia",
                        "age": "22",
                        "bio": "",
                        "interests": [],
                        "_photo_url": "https://images-ssl.gotinder.com/u/userA/photo2.webp?Policy=ccc",
                        "_photo_features": {},
                    },
                    "CURTIR",
                    "CURTIR",
                )

                self.assertEqual(review_id, "")
                self.assertFalse(review_path.exists())
            finally:
                review_queue.REVIEW_PATH = old_review_path
                review_queue.PROFILES_PATH = old_profiles_path

    def test_history_review_updates_existing_training_row_instead_of_duplicating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles = root / "data" / "profiles.csv"
            review_path = root / "data" / "review_queue.csv"
            photo_dir = root / "data" / "photos" / "liked"
            photo_dir.mkdir(parents=True)
            (photo_dir / "abc_Julia_22.jpg").write_bytes(b"photo")
            profiles.parent.mkdir(parents=True, exist_ok=True)
            with profiles.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=dataset.CSV_FIELDNAMES)
                writer.writeheader()
                row = {key: "" for key in dataset.CSV_FIELDNAMES}
                row.update({
                    "name": "Julia",
                    "age": "22",
                    "label": "1",
                    "source": "real",
                    "final_decision": "CURTIR",
                    "photo_features_saved": "1",
                    "photo_has_face": "1",
                    "photo_woman_confidence": "0.9",
                })
                writer.writerow(row)

            old_review_path = review_queue.REVIEW_PATH
            old_review_profiles_path = review_queue.PROFILES_PATH
            old_dataset_profiles_path = dataset.PROFILES_PATH
            old_root = review_queue.ROOT_DIR
            old_photos_dir = review_queue.PHOTOS_DIR
            try:
                review_queue.REVIEW_PATH = review_path
                review_queue.PROFILES_PATH = profiles
                dataset.PROFILES_PATH = profiles
                review_queue.ROOT_DIR = root
                review_queue.PHOTOS_DIR = root / "data" / "photos"

                before_stats = review_queue.history_review_candidate_stats(min_score=0)
                self.assertEqual(before_stats["actionable"], 1)
                self.assertEqual(before_stats["pending_history"], 0)

                result = review_queue.enqueue_history_review_candidates(limit=1, min_score=0)
                self.assertEqual(result["enqueued"], 1)
                after_stats = review_queue.history_review_candidate_stats(min_score=0)
                self.assertEqual(after_stats["actionable"], 0)
                self.assertEqual(after_stats["pending_history"], 1)
                with review_path.open(encoding="utf-8") as f:
                    reviews = list(csv.DictReader(f))
                self.assertEqual(reviews[0]["review_mode"], "history")

                ok = review_queue.apply_review(
                    reviews[0]["review_id"],
                    "NÃO CURTIR",
                    feedback_domain="photo",
                    feedback_reason="foto mudou minha decisão",
                    feedback_intensity="2",
                )
                self.assertTrue(ok)

                with profiles.open(encoding="utf-8") as f:
                    profile_rows = list(csv.DictReader(f))
                self.assertEqual(len(profile_rows), 1)
                self.assertEqual(profile_rows[0]["label"], "0")
                self.assertEqual(profile_rows[0]["source"], "real")
                self.assertEqual(profile_rows[0]["feedback_domain"], "photo")
                self.assertEqual(profile_rows[0]["manual_corrected"], "1")

                post_apply_stats = review_queue.history_review_candidate_stats(min_score=0)
                self.assertEqual(post_apply_stats["actionable"], 0)
                self.assertEqual(post_apply_stats["pending_history"], 0)
                self.assertEqual(review_queue.enqueue_history_review_candidates(limit=1, min_score=0)["enqueued"], 0)

                self.assertTrue(review_queue.apply_review(reviews[0]["review_id"], "NÃO CURTIR"))
                with profiles.open(encoding="utf-8") as f:
                    repeated_rows = list(csv.DictReader(f))
                self.assertEqual(len(repeated_rows), 1)
                self.assertEqual(repeated_rows[0]["label"], "0")

                self.assertTrue(review_queue.undo_review(reviews[0]["review_id"]))
                with profiles.open(encoding="utf-8") as f:
                    restored_rows = list(csv.DictReader(f))
                self.assertEqual(restored_rows[0]["label"], "1")
                self.assertEqual(restored_rows[0]["final_decision"], "CURTIR")
            finally:
                review_queue.REVIEW_PATH = old_review_path
                review_queue.PROFILES_PATH = old_review_profiles_path
                dataset.PROFILES_PATH = old_dataset_profiles_path
                review_queue.ROOT_DIR = old_root
                review_queue.PHOTOS_DIR = old_photos_dir

    def test_history_review_applies_after_non_identity_profile_backfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles = root / "data" / "profiles.csv"
            review_path = root / "data" / "review_queue.csv"
            photo_dir = root / "data" / "photos" / "liked"
            photo_dir.mkdir(parents=True)
            (photo_dir / "abc_Julia_22.jpg").write_bytes(b"photo")
            profiles.parent.mkdir(parents=True, exist_ok=True)
            with profiles.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=dataset.CSV_FIELDNAMES)
                writer.writeheader()
                row = {key: "" for key in dataset.CSV_FIELDNAMES}
                row.update({
                    "name": "Julia",
                    "age": "22",
                    "label": "1",
                    "source": "real",
                    "final_decision": "CURTIR",
                    "interests": "viagem",
                    "photo_features_saved": "1",
                    "photo_has_face": "1",
                    "photo_woman_confidence": "0.9",
                })
                writer.writerow(row)

            old_review_path = review_queue.REVIEW_PATH
            old_review_profiles_path = review_queue.PROFILES_PATH
            old_dataset_profiles_path = dataset.PROFILES_PATH
            old_root = review_queue.ROOT_DIR
            old_photos_dir = review_queue.PHOTOS_DIR
            try:
                review_queue.REVIEW_PATH = review_path
                review_queue.PROFILES_PATH = profiles
                dataset.PROFILES_PATH = profiles
                review_queue.ROOT_DIR = root
                review_queue.PHOTOS_DIR = root / "data" / "photos"

                self.assertEqual(review_queue.enqueue_history_review_candidates(limit=1, min_score=0)["enqueued"], 1)
                with review_path.open(encoding="utf-8") as f:
                    reviews = list(csv.DictReader(f))

                with profiles.open(encoding="utf-8", newline="") as f:
                    rows = list(csv.DictReader(f))
                rows[0]["photo_pose_torso_visibility"] = "0.77"
                with profiles.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=dataset.CSV_FIELDNAMES)
                    writer.writeheader()
                    writer.writerows(rows)

                self.assertTrue(review_queue.apply_review(
                    reviews[0]["review_id"],
                    "CURTIR",
                    feedback_domain="other",
                    feedback_reason="confirmado",
                    feedback_intensity="1",
                ))
                with profiles.open(encoding="utf-8") as f:
                    updated = list(csv.DictReader(f))
                self.assertEqual(updated[0]["label"], "1")
                self.assertEqual(updated[0]["feedback_reason"], "confirmado")
            finally:
                review_queue.REVIEW_PATH = old_review_path
                review_queue.PROFILES_PATH = old_review_profiles_path
                dataset.PROFILES_PATH = old_dataset_profiles_path
                review_queue.ROOT_DIR = old_root
                review_queue.PHOTOS_DIR = old_photos_dir

    def test_history_review_ignores_profiles_already_handled_by_normal_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profiles = root / "data" / "profiles.csv"
            review_path = root / "data" / "review_queue.csv"
            photo_dir = root / "data" / "photos" / "liked"
            photo_dir.mkdir(parents=True)
            photo_rel = "data/photos/liked/abc_Julia_22.jpg"
            (root / photo_rel).write_bytes(b"photo")
            profiles.parent.mkdir(parents=True, exist_ok=True)
            row = {key: "" for key in dataset.CSV_FIELDNAMES}
            row.update({
                "name": "Julia",
                "age": "22",
                "label": "1",
                "source": "real",
                "final_decision": "CURTIR",
                "photo_features_saved": "1",
            })
            with profiles.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=dataset.CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerow(row)

            normal_review = {key: "" for key in review_queue.REVIEW_FIELDNAMES}
            normal_review.update({
                "review_id": "normal-review",
                "created_at": "2026-05-13T09:00:00",
                "review_status": "reviewed",
                "reviewed_at": "2026-05-13T09:01:00",
                "photo_path": photo_rel,
                "original_label": "CURTIR",
                "review_mode": "auto",
                "name": "Julia",
                "age": "22",
                "final_decision": "CURTIR",
                "label": "1",
                "source": "auto_review",
                "feedback_domain": "photo",
                "feedback_reason": "review normal ja confirmou",
            })
            with review_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=review_queue.REVIEW_FIELDNAMES)
                writer.writeheader()
                writer.writerow(normal_review)

            old_review_path = review_queue.REVIEW_PATH
            old_review_profiles_path = review_queue.PROFILES_PATH
            old_dataset_profiles_path = dataset.PROFILES_PATH
            old_root = review_queue.ROOT_DIR
            old_photos_dir = review_queue.PHOTOS_DIR
            try:
                review_queue.REVIEW_PATH = review_path
                review_queue.PROFILES_PATH = profiles
                dataset.PROFILES_PATH = profiles
                review_queue.ROOT_DIR = root
                review_queue.PHOTOS_DIR = root / "data" / "photos"

                stats = review_queue.history_review_candidate_stats(min_score=0)
                self.assertEqual(stats["actionable"], 0)
                self.assertEqual(stats["skipped_existing"], 1)
                self.assertEqual(review_queue.enqueue_history_review_candidates(limit=1, min_score=0)["enqueued"], 0)

                history_row = review_queue._history_review_row(
                    row,
                    0,
                    dataset.profile_row_signature(row),
                    photo_rel,
                    10,
                )
                with review_path.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=review_queue.REVIEW_FIELDNAMES)
                    writer.writerow({key: history_row.get(key, "") for key in review_queue.REVIEW_FIELDNAMES})

                self.assertEqual(review_queue.skip_duplicate_history_reviews(), 1)
                with review_path.open(encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(rows[1]["review_status"], "skipped")
                self.assertIn("fila normal", rows[1]["feedback_reason"])
            finally:
                review_queue.REVIEW_PATH = old_review_path
                review_queue.PROFILES_PATH = old_review_profiles_path
                dataset.PROFILES_PATH = old_dataset_profiles_path
                review_queue.ROOT_DIR = old_root
                review_queue.PHOTOS_DIR = old_photos_dir

    def test_body_photo_review_dedupes_same_profile_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo_dir = root / "data" / "photos" / "liked"
            photo_dir.mkdir(parents=True)
            (photo_dir / "abc_Julia_22_body.jpg").write_bytes(b"newer")
            (photo_dir / "def_Julia_22_body.jpg").write_bytes(b"older")
            (photo_dir / "ghi_Maria_23_body.jpg").write_bytes(b"other")

            profiles = root / "data" / "profiles.csv"
            profiles.parent.mkdir(parents=True, exist_ok=True)
            with profiles.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=dataset.CSV_FIELDNAMES)
                writer.writeheader()
                for name, age in (("Julia", "22"), ("Maria", "23")):
                    row = {key: "" for key in dataset.CSV_FIELDNAMES}
                    row.update({
                        "name": name,
                        "age": age,
                        "source": "real",
                        "label": "1",
                        "photo_body_visible": "0.7",
                        "photo_body_signal_quality": "0.7",
                        "photo_pose_shoulder_width": "0.18",
                        "photo_pose_hip_width": "0.16",
                        "photo_pose_torso_visibility": "0.45",
                    })
                    writer.writerow(row)

            old_root = photo_deep_feedback.ROOT_DIR
            old_profiles_path = dataset.PROFILES_PATH
            try:
                photo_deep_feedback.ROOT_DIR = root
                dataset.PROFILES_PATH = profiles
                paths = photo_deep_feedback.list_saved_photo_paths(None, body_only=True)
                self.assertEqual(len(paths), 2)
                self.assertEqual(sum("Julia_22_body" in path for path in paths), 1)
            finally:
                photo_deep_feedback.ROOT_DIR = old_root
                dataset.PROFILES_PATH = old_profiles_path

    def test_body_train_requires_real_body_measurements(self):
        face_only = {
            "photo_body_visible": "0.1",
            "photo_body_signal_quality": "0.05",
            "photo_pose_shoulder_width": "0",
            "photo_pose_hip_width": "0",
            "photo_pose_torso_visibility": "0",
        }
        measured = {
            "photo_body_visible": "0.7",
            "photo_body_signal_quality": "0.7",
            "photo_pose_shoulder_width": "0.18",
            "photo_pose_hip_width": "0.16",
            "photo_pose_torso_visibility": "0.45",
            "photo_pose_body_coverage": "0.35",
        }
        self.assertLess(_body_measurement_strength(face_only), 0.45)
        self.assertGreaterEqual(_body_measurement_strength(measured), 0.45)

    def test_body_measurement_rejects_face_only_false_pose(self):
        face_only_false_pose = {
            "photo_body_visible": "0",
            "photo_body_signal_quality": "0",
            "photo_body_full_length": "0",
            "photo_body_upper_length": "0",
            "photo_body_closeup": "1",
            "photo_pose_shoulder_width": "0.55",
            "photo_pose_hip_width": "0.30",
            "photo_pose_torso_visibility": "0.8",
            "photo_pose_body_coverage": "0.75",
        }

        self.assertLess(_body_measurement_strength(face_only_false_pose), 0.45)

    def test_analyze_photos_does_not_save_face_only_as_body(self):
        def result(**updates):
            data = photo_features._default_features()
            data.update({
                "photo_has_face": 1,
                "photo_woman_confidence": 0.9,
                "photo_faces_ratio": 1.0,
                "photo_gender_certainty": 0.8,
                "_face_confidence": 0.9,
            })
            data.update(updates)
            return data

        old_analyze_photo = photo_features.analyze_photo
        old_get_photos_config = photo_features.get_photos_config
        try:
            photo_features.get_photos_config = lambda: {
                "analysis_parallel_workers": 1,
                "enable_face_embedding": False,
                "embedding_strategy": "best_photo",
                "face_save_min_confidence": 0.5,
                "body_save_min_quality": 0.25,
                "body_save_min_strength": 0.45,
            }
            fake_results = {
                "face-a": result(
                    photo_body_visible=0.78,
                    photo_body_upper_length=0.70,
                    photo_body_signal_quality=0.78,
                    photo_body_closeup=0.92,
                    photo_image_sharpness=0.8,
                    photo_image_brightness=0.55,
                ),
                "face-b": result(
                    photo_body_visible=0.12,
                    photo_body_signal_quality=0.12,
                    photo_body_closeup=0.95,
                ),
            }
            photo_features.analyze_photo = lambda url, profile_age=0: dict(fake_results[url])

            aggregated = photo_features.analyze_photos(["face-a", "face-b"], max_photos=2)

            self.assertEqual(aggregated["_best_body_photo_url"], "")
            self.assertEqual(aggregated["_best_body_photo_score"], 0.0)
        finally:
            photo_features.analyze_photo = old_analyze_photo
            photo_features.get_photos_config = old_get_photos_config

    def test_analyze_photos_allows_same_photo_as_face_and_body_when_body_is_real(self):
        data = photo_features._default_features()
        data.update({
            "photo_has_face": 1,
            "photo_woman_confidence": 0.9,
            "photo_faces_ratio": 1.0,
            "photo_gender_certainty": 0.8,
            "_face_confidence": 0.9,
            "photo_body_visible": 0.9,
            "photo_body_full_length": 0.25,
            "photo_body_upper_length": 0.75,
            "photo_body_signal_quality": 0.85,
            "photo_body_closeup": 0.45,
            "photo_pose_shoulder_width": 0.18,
            "photo_pose_hip_width": 0.14,
            "photo_pose_torso_visibility": 0.7,
            "photo_pose_body_coverage": 0.4,
        })

        old_analyze_photo = photo_features.analyze_photo
        old_get_photos_config = photo_features.get_photos_config
        try:
            photo_features.get_photos_config = lambda: {
                "analysis_parallel_workers": 1,
                "enable_face_embedding": False,
                "embedding_strategy": "best_photo",
                "face_save_min_confidence": 0.5,
                "body_save_min_quality": 0.25,
                "body_save_min_strength": 0.45,
            }
            photo_features.analyze_photo = lambda url, profile_age=0: dict(data)

            aggregated = photo_features.analyze_photos(["same-photo"], max_photos=1)

            self.assertEqual(aggregated["_best_face_photo_url"], "same-photo")
            self.assertEqual(aggregated["_best_body_photo_url"], "same-photo")
            self.assertGreaterEqual(aggregated["_best_body_photo_score"], 0.45)
        finally:
            photo_features.analyze_photo = old_analyze_photo
            photo_features.get_photos_config = old_get_photos_config

    def test_profile_parser_keeps_richer_recs_metadata_and_descriptors(self):
        descriptors = extract_descriptors([
            {
                "id": "de_38",
                "type": "multi_selection_set",
                "section_name": "Tipo de relacionamento",
                "choice_selections": [{"name": "Monogamia"}, {"name": "Algo casual"}],
            },
            {
                "id": "de_30",
                "type": "measurement",
                "section_name": "Altura",
                "measurable_selection": {"value": 160, "unit_of_measure": "cm"},
            },
        ])

        self.assertEqual(descriptors["Tipo de relacionamento"], "Monogamia, Algo casual")
        self.assertEqual(descriptors["Altura"], "160 cm")

        profile = parse_profile({
            "type": "user",
            "content_hash": "hash-1",
            "s_number": 123,
            "distance_mi": 10,
            "user": {
                "_id": "abc",
                "name": "Julia",
                "birth_date": "2004-01-01T00:00:00.000Z",
                "selected_descriptors": [],
                "relationship_intent": {"title_text": "Tô procurando", "body_text": "Nada sério"},
                "online_now": True,
                "recently_active": True,
                "photos": [],
            },
        })

        self.assertEqual(profile["_content_hash"], "hash-1")
        self.assertEqual(profile["_s_number"], 123)
        self.assertEqual(profile["_descriptors"]["Tô procurando"], "Nada sério")
        self.assertTrue(profile["_online_now"])

    def test_profile_parser_extracts_basic_info_and_pronouns(self):
        profile = parse_profile({
            "type": "user",
            "user": {
                "_id": "abc",
                "name": "Julia",
                "birth_date": "2004-01-01T00:00:00.000Z",
                "selected_descriptors": [
                    {
                        "section_name": "Pronomes",
                        "choice_selections": [{"name": "ela/dela"}],
                    }
                ],
                "profile_detail_content": {
                    "page_content_id": "essentials",
                    "title": "Informações básicas",
                    "items": [
                        {"name": "Identidade de gênero", "value": "Mulher trans"},
                    ],
                },
                "photos": [],
            },
        })

        self.assertEqual(profile["_descriptors"]["Pronomes"], "ela/dela")
        self.assertIn("Informações básicas: Identidade de gênero", profile["_descriptors"])
        self.assertEqual(profile["_descriptors"]["Informações básicas: Identidade de gênero"], "Mulher trans")

    def test_recent_swipe_cache_matches_id_and_name_age(self):
        state.clear_recent_swipes()
        try:
            state.mark_recent_swipe("abc", "Julia", 22, action="like", source="test", status=200)
            by_id, record = state.is_recently_swiped(tinder_id="abc")
            self.assertTrue(by_id)
            self.assertEqual(record["action"], "like")

            by_name, _ = state.is_recently_swiped(name="Júlia", age=22)
            self.assertTrue(by_name)
        finally:
            state.clear_recent_swipes()

    def test_navigation_request_state_round_trip(self):
        state.clear_navigation_request()
        try:
            generation = state.request_navigation(
                "https://tinder.com/app/recs",
                "paywall_url: voltando ao swipe",
            )
            pending, target, reason, requested_at, stored_generation = state.peek_navigation_request()

            self.assertTrue(pending)
            self.assertEqual(target, "https://tinder.com/app/recs")
            self.assertIn("paywall", reason)
            self.assertGreater(requested_at, 0)
            self.assertEqual(stored_generation, generation)
        finally:
            state.clear_navigation_request()

    def test_resource_guard_detects_memory_pressure_by_percent_or_available_mb(self):
        cfg = {"swiper": {"mem_pressure_threshold_percent": 88, "min_mem_available_mb": 1200}}

        pressure, reason, _ = is_memory_pressure(
            cfg,
            snapshot={"mem_used_pct": 87.0, "mem_avail_mb": 900.0},
        )
        self.assertTrue(pressure)
        self.assertIn("mem_avail_mb", reason)

        pressure, reason, _ = is_memory_pressure(
            cfg,
            snapshot={"mem_used_pct": 89.0, "mem_avail_mb": 2000.0},
        )
        self.assertTrue(pressure)
        self.assertIn("mem_used_pct", reason)

        pressure, _, _ = is_memory_pressure(
            cfg,
            snapshot={"mem_used_pct": 60.0, "mem_avail_mb": 5000.0},
        )
        self.assertFalse(pressure)

    def test_memory_pressure_photo_features_are_neutral_and_non_blocking(self):
        features = memory_pressure_photo_features("mem_avail_mb=900<=1200")

        self.assertTrue(features["_analysis_failed"])
        self.assertTrue(features["_analysis_skipped"])
        self.assertEqual(features["photo_has_face"], 1.0)
        self.assertEqual(features["photo_woman_confidence"], 0.5)

    def test_super_like_balance_state_uses_network_balance(self):
        state.clear_active_profiles()
        state.set_super_like_balance({
            "remaining": 0,
            "alc_remaining": 0,
            "new_alc_remaining": 0,
            "available": False,
            "source": "test",
        })

        balance, age = state.get_super_like_balance(10)

        self.assertIsNotNone(balance)
        self.assertLess(age, 10)
        self.assertFalse(balance["available"])

    def test_pass_correction_gets_extra_training_weight(self):
        import pandas as pd

        df = pd.DataFrame([
            {
                "source": "real",
                "feedback_domain": "photo",
                "feedback_intensity": "2",
                "manual_corrected": "1",
                "label": "0",
                "ai_decision": "CURTIR",
                "feedback_details": "{}",
            },
            {
                "source": "real",
                "feedback_domain": "photo",
                "feedback_intensity": "2",
                "manual_corrected": "1",
                "label": "1",
                "ai_decision": "NÃO CURTIR",
                "feedback_details": "{}",
            },
        ])
        cfg = {
            "model": {
                "feedback_weights": {
                    "enabled": True,
                    "manual_correction_multiplier": 1.25,
                    "pass_correction_multiplier": 2.25,
                    "max_sample_weight": 100,
                }
            }
        }

        weights = model_training._sample_weights(df, "photo", cfg)

        self.assertGreater(weights[0], weights[1])

    def test_photo_deep_feedback_gets_visual_training_weight(self):
        import pandas as pd

        df = pd.DataFrame([
            {
                "source": "photo_deep",
                "feedback_domain": "photo",
                "feedback_intensity": "2",
                "manual_corrected": "1",
                "label": "0",
                "ai_decision": "CURTIR",
                "feedback_details": "{}",
            },
            {
                "source": "photo_deep",
                "feedback_domain": "photo",
                "feedback_intensity": "2",
                "manual_corrected": "",
                "label": "1",
                "ai_decision": "",
                "feedback_details": "{}",
            },
        ])
        cfg = {"model": {"feedback_weights": {"enabled": True, "photo_deep_weight": 1.6, "max_sample_weight": 100}}}

        photo_weights = model_training._sample_weights(df, "photo", cfg)
        text_weights = model_training._sample_weights(df, "text", cfg)

        self.assertGreater(photo_weights[0], 1.6)
        self.assertLess(text_weights[0], photo_weights[0])

    def test_semantic_embedding_cache_hit_miss_and_nan_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "semantic_cache.json"
            key = "url:/u/test/photo.webp"
            cache_path.write_text(
                '{"url:/u/test/photo.webp":{"model_name":"test-clip","embedding":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6],"device":"cpu"}}',
                encoding="utf-8",
            )

            old_config = photo_semantic_embeddings.get_photos_config
            old_cache = photo_semantic_embeddings._CACHE_DATA
            try:
                photo_semantic_embeddings._CACHE_DATA = None
                photo_semantic_embeddings.get_photos_config = lambda: {
                    "semantic_embedding": {
                        "enabled": True,
                        "model_name": "test-clip",
                        "cache_path": str(cache_path),
                    }
                }
                photo_semantic_embeddings.cache_stats(reset=True)
                emb = photo_semantic_embeddings.get_cached_embedding(key)
                miss = photo_semantic_embeddings.get_cached_embedding("url:/missing")
                stats = photo_semantic_embeddings.cache_stats()
                nan_features = photo_semantic_embeddings.apply_to_embedding(None, None)

                self.assertIsNotNone(emb)
                self.assertIsNone(miss)
                self.assertEqual(stats["hits"], 1)
                self.assertEqual(stats["misses"], 1)
                self.assertTrue(all(value != value for value in nan_features.values()))
            finally:
                photo_semantic_embeddings.get_photos_config = old_config
                photo_semantic_embeddings._CACHE_DATA = old_cache

    def test_semantic_embedding_batch_for_paths_uses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "semantic_cache.json"
            img1 = Path(tmp) / "one.jpg"
            img2 = Path(tmp) / "two.jpg"
            try:
                from PIL import Image
            except Exception as exc:
                self.skipTest(f"PIL indisponivel: {exc}")
            Image.new("RGB", (8, 8), (255, 0, 0)).save(img1)
            Image.new("RGB", (8, 8), (0, 255, 0)).save(img2)

            class FakeModel:
                def encode(self, images, batch_size=1, **_kwargs):
                    self.last_batch_size = batch_size
                    return [[float(i + 1)] * 16 for i, _ in enumerate(images)]

            fake_model = FakeModel()
            old_config = photo_semantic_embeddings.get_photos_config
            old_cache = photo_semantic_embeddings._CACHE_DATA
            old_get_model = photo_semantic_embeddings._get_model
            try:
                photo_semantic_embeddings._CACHE_DATA = None
                photo_semantic_embeddings.get_photos_config = lambda: {
                    "semantic_embedding": {
                        "enabled": True,
                        "model_name": "test-clip-batch",
                        "cache_path": str(cache_path),
                        "batch_size": 2,
                        "batch_timeout_seconds": 0,
                    }
                }
                photo_semantic_embeddings._get_model = lambda: (fake_model, "cuda")
                first = photo_semantic_embeddings.embeddings_for_paths([img1, img2])
                self.assertEqual(len(first), 2)
                self.assertEqual(fake_model.last_batch_size, 2)

                cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
                self.assertEqual(len(cache_data), 2)

                photo_semantic_embeddings.cache_stats(reset=True)
                second = photo_semantic_embeddings.embeddings_for_paths([img1, img2])
                stats = photo_semantic_embeddings.cache_stats()
                self.assertEqual(len(second), 2)
                self.assertEqual(stats["hits"], 2)
            finally:
                photo_semantic_embeddings.get_photos_config = old_config
                photo_semantic_embeddings._CACHE_DATA = old_cache
                photo_semantic_embeddings._get_model = old_get_model

    def test_extract_features_includes_semantic_photo_features(self):
        profile = {
            "name": "Ana",
            "age": 22,
            "bio": "",
            "interests": [],
            "_photo_features": {
                "photo_semantic_embedding_saved": 1,
                "photo_carousel_useful_count": 3,
                "photo_carousel_duplicate_score": 0.8,
                "photo_carousel_visual_diversity": 0.2,
                "photo_clip_pc_01": 0.42,
            },
        }
        cfg = {
            "preferences": {
                "age_range": [18, 27],
                "preferred_interests": [],
                "positive_bio_keywords": [],
                "negative_bio_keywords": [],
                "disliked_names": [],
            }
        }

        features = extract_features(profile, cfg)

        self.assertEqual(features["photo_semantic_embedding_saved"], 1.0)
        self.assertEqual(features["photo_carousel_useful_count"], 3.0)
        self.assertEqual(features["photo_clip_pc_01"], 0.42)
        self.assertTrue(all(name in features for name in SEMANTIC_EMBEDDING_FEATURE_NAMES))

    def test_review_ui_hides_body_carousel_when_body_file_duplicates_face_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo_dir = root / "data" / "photos" / "liked"
            photo_dir.mkdir(parents=True)
            face = photo_dir / "abc_Ana_22.jpg"
            body = photo_dir / "abc_Ana_22_body.jpg"
            face.write_bytes(b"same-image")
            body.write_bytes(b"same-image")
            row = {
                "photo_path": "data/photos/liked/abc_Ana_22.jpg",
                "photo_body_visible": "0.9",
                "photo_body_signal_quality": "0.9",
                "photo_body_upper_length": "0.8",
                "photo_pose_shoulder_width": "0.18",
                "photo_pose_hip_width": "0.14",
                "photo_pose_torso_visibility": "0.7",
            }
            old_root = review_ui.ROOT_DIR
            try:
                review_ui.ROOT_DIR = root
                self.assertEqual(review_ui._body_photo_path(row), "")
            finally:
                review_ui.ROOT_DIR = old_root

    def test_photo_storage_saves_body_alias_when_body_matches_main_photo(self):
        calls = []
        old_download = photo_storage.download_photo_async
        old_config = photo_storage.get_photos_config

        try:
            photo_storage.get_photos_config = lambda: {"save_face_body_pair": True}
            photo_storage.download_photo_async = (
                lambda url, tinder_id, name, age, decision, reason="", role="": calls.append(
                    {
                        "url": url,
                        "role": role,
                        "reason": reason,
                    }
                )
            )

            profile = {
                "name": "Ana",
                "age": 22,
                "_tinder_id": "abc123",
                "_photo_url": "https://img.example/fallback.jpg",
                "_photo_features": {
                    "_review_photo_url": "https://img.example/body.jpg",
                    "_best_face_photo_url": "https://img.example/body.jpg",
                    "_best_body_photo_url": "https://img.example/body.jpg",
                    "photo_body_visible": 0.9,
                    "photo_body_full_length": 0.25,
                    "photo_body_upper_length": 0.75,
                    "photo_body_signal_quality": 0.85,
                    "photo_body_closeup": 0.45,
                    "photo_pose_shoulder_width": 0.18,
                    "photo_pose_hip_width": 0.14,
                    "photo_pose_torso_visibility": 0.7,
                    "photo_pose_body_coverage": 0.4,
                },
            }

            photo_storage.download_profile_photos_async(profile, "CURTIR", "Decisao")

            self.assertEqual([call["role"] for call in calls], ["", "body"])
            self.assertEqual(calls[0]["url"], calls[1]["url"])
            self.assertIn("Foto salva para: analise de corpo", calls[1]["reason"])
        finally:
            photo_storage.download_photo_async = old_download
            photo_storage.get_photos_config = old_config

    def test_photo_storage_skips_body_alias_when_body_signal_is_weak(self):
        calls = []
        old_download = photo_storage.download_photo_async
        old_config = photo_storage.get_photos_config

        try:
            photo_storage.get_photos_config = lambda: {"save_face_body_pair": True}
            photo_storage.download_photo_async = (
                lambda url, tinder_id, name, age, decision, reason="", role="": calls.append(
                    {
                        "url": url,
                        "role": role,
                    }
                )
            )

            profile = {
                "name": "Ana",
                "age": 22,
                "_tinder_id": "abc123",
                "_photo_url": "https://img.example/fallback.jpg",
                "_photo_features": {
                    "_review_photo_url": "https://img.example/face.jpg",
                    "_best_face_photo_url": "https://img.example/face.jpg",
                    "_best_body_photo_url": "https://img.example/face.jpg",
                    "photo_body_visible": 0.0,
                    "photo_body_signal_quality": 0.0,
                    "photo_body_closeup": 1.0,
                    "photo_pose_shoulder_width": 0.55,
                    "photo_pose_hip_width": 0.30,
                    "photo_pose_torso_visibility": 0.8,
                    "photo_pose_body_coverage": 0.75,
                },
            }

            photo_storage.download_profile_photos_async(profile, "CURTIR", "Decisao")

            self.assertEqual([call["role"] for call in calls], [""])
            self.assertEqual(profile["_body_photo_url"], "")
        finally:
            photo_storage.download_photo_async = old_download
            photo_storage.get_photos_config = old_config


if __name__ == "__main__":
    unittest.main()
