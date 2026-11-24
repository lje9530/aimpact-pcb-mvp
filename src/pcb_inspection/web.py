from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import joblib
import numpy as np
from flask import Flask, Response, g, jsonify, render_template, request, send_file
from openpyxl import Workbook
from PIL import UnidentifiedImageError

from .imports import FIELDS, REQUIRED, normalized, read_table, suggest
from .vision import CLASSES, MODEL_VERSION, png, predict, read_image


SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
 id INTEGER PRIMARY KEY, external_id TEXT NOT NULL, supplier TEXT NOT NULL, product TEXT NOT NULL,
 inspected_at TEXT NOT NULL, process TEXT NOT NULL, batch TEXT NOT NULL,
 reported_result TEXT NOT NULL, reported_type TEXT NOT NULL,
 image_file TEXT NOT NULL, reference_file TEXT NOT NULL, data_source TEXT NOT NULL,
 test_image BLOB NOT NULL, reference_image BLOB, width INTEGER NOT NULL, height INTEGER NOT NULL,
 ai_result TEXT NOT NULL, ai_types TEXT NOT NULL, score REAL, signals TEXT NOT NULL,
 ms REAL, model_version TEXT NOT NULL, roi TEXT, ground_truth TEXT NOT NULL,
 UNIQUE(supplier, external_id)
);
CREATE TABLE IF NOT EXISTS reviews (
 id INTEGER PRIMARY KEY, inspection_id INTEGER NOT NULL REFERENCES inspections(id),
 revision INTEGER NOT NULL, result TEXT NOT NULL, types TEXT NOT NULL, reviewer TEXT NOT NULL,
 note TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(inspection_id, revision)
);
CREATE TABLE IF NOT EXISTS templates (
 supplier TEXT PRIMARY KEY, mapping TEXT NOT NULL
);
"""

SUMMARY = """
SELECT i.id, i.external_id, i.supplier, i.product, i.inspected_at, i.process, i.batch,
       i.reported_result, i.reported_type, i.image_file, i.reference_file, i.data_source,
       i.width, i.height, i.ai_result, i.ai_types, i.score, i.ms, i.model_version, i.roi,
       r.revision, r.result AS final_result, r.types AS final_types, r.reviewer, r.note,
       r.created_at AS reviewed_at
FROM inspections i
LEFT JOIN reviews r ON r.id = (SELECT id FROM reviews WHERE inspection_id = i.id ORDER BY revision DESC LIMIT 1)
"""


def db():
    if "db" not in g:
        g.db = sqlite3.connect(g.database_path, timeout=8)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys=ON")
    return g.db


def serialize(row):
    result = dict(row)
    for field in ("ai_types", "final_types", "types", "signals", "roi", "ground_truth"):
        if result.get(field) is not None:
            result[field] = json.loads(result[field])
    return result


def attach_images(record, test_bytes, ref_bytes, artifact, roi=None):
    test = read_image(test_bytes)
    reference = read_image(ref_bytes) if ref_bytes is not None else None
    ai = predict(artifact["model"], test, reference, roi, artifact["threshold"])
    return {**record, "test_image": png(test), "reference_image": png(reference) if reference is not None else None,
            "width": test.shape[1], "height": test.shape[0], "ai_result": ai["result"],
            "ai_types": json.dumps(ai["types"]), "score": ai["score"], "signals": json.dumps(ai["signals"]),
            "ms": ai["ms"], "model_version": artifact["version"], "roi": json.dumps(roi),
            "ground_truth": json.dumps(record.get("ground_truth", []))}


def insert(record):
    columns = ("external_id supplier product inspected_at process batch reported_result reported_type "
               "image_file reference_file data_source test_image reference_image width height ai_result "
               "ai_types score signals ms model_version roi ground_truth").split()
    sql = f"INSERT INTO inspections ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})"
    cursor = db().execute(sql + " ON CONFLICT(supplier, external_id) DO NOTHING", [record[key] for key in columns])
    return cursor.lastrowid if cursor.rowcount else None


def review_entry(record_id, revision, result, kinds, reviewer, note):
    db().execute("INSERT INTO reviews (inspection_id,revision,result,types,reviewer,note,created_at) VALUES (?,?,?,?,?,?,?)",
                 (record_id, revision, result, json.dumps(kinds), reviewer, note, datetime.now(timezone.utc).isoformat(timespec="seconds")))


def seed(assets, artifact):
    manifest = json.loads((assets / "samples.json").read_text(encoding="utf-8"))
    if db().execute("SELECT 1 FROM inspections LIMIT 1").fetchone():
        return 0
    prepared = [attach_images(row, (assets / "images" / row["image_file"]).read_bytes(),
                              (assets / "images" / row["reference_file"]).read_bytes(), artifact) for row in manifest]
    inserted = 0
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM inspections LIMIT 1").fetchone():
            return 0
        for index, row in enumerate(prepared):
            record_id = insert(row)
            inserted += int(bool(record_id))
            if index in (0, 3, 4, 5):
                kinds = sorted({mark[4] for mark in manifest[index]["ground_truth"]})
                review_entry(record_id, 1, "NG" if kinds else "OK", kinds, "데모 담당자", "공개 라벨 기반 시연용 검토 기록")
    return inserted


def parse_upload():
    metadata = request.files.get("metadata")
    if metadata is None or not metadata.filename:
        raise ValueError("검사 기록 CSV 또는 XLSX 파일을 선택하세요.")
    return read_table(metadata.read(), metadata.filename)


def create_app(database_path: Path, assets: Path, load_demo=True):
    artifact = joblib.load(assets / "model.joblib")
    if artifact["version"] != MODEL_VERSION:
        raise RuntimeError("데모 모델 버전이 다릅니다. 데모 자산을 다시 생성하세요.")
    cv2.setNumThreads(1)
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=48 * 1024 * 1024, MAX_FORM_PARTS=1100)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    @app.before_request
    def set_db_path():
        g.database_path = database_path

    @app.teardown_appcontext
    def close_db(_error):
        connection = g.pop("db", None)
        if connection is not None:
            connection.close()

    @app.errorhandler(ValueError)
    @app.errorhandler(UnidentifiedImageError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error="파일이 너무 많거나 전체 크기가 48 MB를 넘습니다."), 413

    with app.app_context():
        set_db_path()
        db().executescript(SCHEMA)
        if load_demo:
            seed(assets, artifact)

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/api/inspections")
    def inspections():
        rows = db().execute(SUMMARY + " ORDER BY i.inspected_at DESC, i.id DESC")
        return jsonify(items=[serialize(row) for row in rows], classes=CLASSES)

    @app.get("/api/inspections/<int:item_id>")
    def detail(item_id):
        row = db().execute(SUMMARY + " WHERE i.id=?", (item_id,)).fetchone()
        if row is None:
            return jsonify(error="검사 기록을 찾을 수 없습니다."), 404
        evidence = db().execute("SELECT signals, ground_truth FROM inspections WHERE id=?", (item_id,)).fetchone()
        history = db().execute("SELECT * FROM reviews WHERE inspection_id=? ORDER BY revision DESC", (item_id,))
        return jsonify(**serialize(row), **serialize(evidence), history=[serialize(entry) for entry in history])

    @app.get("/api/inspections/<int:item_id>/image/<variant>")
    def image(item_id, variant):
        columns = {"test": "test_image", "reference": "reference_image"}
        if variant not in columns:
            return "", 404
        row = db().execute(f"SELECT {columns[variant]} FROM inspections WHERE id=?", (item_id,)).fetchone()
        if row is None or row[0] is None:
            return "", 404
        return Response(row[0], mimetype="image/png")

    @app.get("/api/inspections/<int:item_id>/difference")
    def difference(item_id):
        pair = db().execute("SELECT test_image, reference_image FROM inspections WHERE id=?", (item_id,)).fetchone()
        if pair is None or pair[1] is None:
            return "", 404
        test, ref = (read_image(content) for content in pair)
        test_mask, ref_mask = (cv2.threshold(image, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] for image in (test, ref))
        changed = np.abs(test_mask.astype(np.int8) - ref_mask)
        halo = cv2.GaussianBlur(cv2.dilate(changed.astype(np.float32), np.ones((5, 5), np.uint8)), (0, 0), 2.2)
        overlay = np.zeros((*test.shape, 4), dtype=np.uint8)
        overlay[:, :, :3] = (34, 104, 255)
        overlay[:, :, 3] = np.clip(changed*160 + halo*115, 0, 255).astype(np.uint8)
        return Response(png(overlay), mimetype="image/png")

    @app.post("/api/inspections/<int:item_id>/review")
    def review(item_id):
        payload = request.get_json()
        if not isinstance(payload, dict):
            raise ValueError("검토 입력을 확인하세요.")
        result, kinds, revision = payload.get("result"), payload.get("types"), payload.get("revision")
        reviewer, note = str(payload.get("reviewer") or "").strip(), str(payload.get("note") or "").strip()
        if result not in ("OK", "NG") or not reviewer or len(reviewer) > 60 or len(note) > 500:
            raise ValueError("판정, 담당자 이름, 메모 길이를 확인하세요.")
        if not isinstance(kinds, list) or any(type(kind) is not int or kind not in CLASSES or kind == 0 for kind in kinds):
            raise ValueError("불량유형 값이 올바르지 않습니다.")
        if result == "NG" and not kinds:
            raise ValueError("불량 확정 시 유형을 하나 이상 선택하세요.")
        if type(revision) is not int or revision < 0:
            raise ValueError("판정 이력 버전을 확인하세요.")
        with db() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(SUMMARY + " WHERE i.id=?", (item_id,)).fetchone()
            if current is None:
                return jsonify(error="검사 기록을 찾을 수 없습니다."), 404
            latest = current["revision"] or 0
            if revision != latest:
                return jsonify(error="다른 담당자가 이 기록을 먼저 수정했습니다. 새로고침 후 다시 검토하세요."), 409
            review_entry(item_id, latest + 1, result, sorted(set(kinds)) if result == "NG" else [], reviewer, note)
        return detail(item_id)

    @app.post("/api/import/preview")
    def preview():
        headers, rows = parse_upload()
        supplier = request.form.get("supplier", "").strip()
        saved = db().execute("SELECT mapping FROM templates WHERE supplier=?", (supplier,)).fetchone()
        mapping = suggest(headers)
        if saved:
            mapping.update({key: value for key, value in json.loads(saved[0]).items() if value in headers})
        return jsonify(headers=headers, count=len(rows), first=rows[0][1], mapping=mapping,
                       fields={key: aliases[0] for key, aliases in FIELDS.items()}, required=REQUIRED, template=bool(saved))

    @app.post("/api/import")
    def import_data():
        headers, rows = parse_upload()
        supplier = request.form.get("supplier", "").strip()
        mapping = json.loads(request.form.get("mapping", "{}"))
        if not supplier or len(supplier) > 60 or not isinstance(mapping, dict) or any(mapping.get(key) not in headers for key in REQUIRED):
            raise ValueError("협력사 이름과 필수 항목의 매핑을 확인하세요.")
        if any(key not in FIELDS or not isinstance(value, str) or (value and value not in headers) for key, value in mapping.items()):
            raise ValueError("공통 항목 매핑을 확인하세요.")
        roi = json.loads(request.form.get("roi", "null"))
        if roi is not None and (not isinstance(roi, list) or len(roi) != 4 or any(type(value) not in (int, float) or not 0 <= value <= 1 for value in roi)):
            raise ValueError("ROI는 0~1 범위의 x, y, 너비, 높이 값 네 개가 필요합니다.")
        files = {}
        for file in request.files.getlist("images"):
            filename = (file.filename or "").replace("\\", "/").split("/")[-1]
            if filename in files or not filename:
                raise ValueError("이미지 파일명이 중복되었거나 비어 있습니다.")
            files[filename] = file.read()
        if not files:
            raise ValueError("검사 영상 파일을 선택하세요.")
        prepared = []
        for index, row in rows:
            try:
                record = normalized(row, mapping)
                test_name, ref_name = record["image_file"], record["reference_file"]
                if test_name not in files or (ref_name and ref_name not in files):
                    raise ValueError("CSV·엑셀의 검사/기준 영상 파일명에 해당하는 이미지가 없습니다.")
                if not record["external_id"]:
                    identity = "\n".join(record[name] for name in ("product", "inspected_at", "process", "batch", "image_file"))
                    record["external_id"] = hashlib.sha256(identity.encode()).hexdigest()[:12]
                record.update(supplier=supplier, data_source="협력사 업로드")
                prepared.append(attach_images(record, files[test_name], files.get(ref_name), artifact, roi))
            except (ValueError, UnidentifiedImageError, cv2.error) as exc:
                raise ValueError(f"{index}행 · {exc}") from exc
        with db() as conn:
            ids = [item_id for item_id in map(insert, prepared) if item_id is not None]
            if request.form.get("save_template") == "1":
                conn.execute("INSERT INTO templates VALUES (?,?) ON CONFLICT(supplier) DO UPDATE SET mapping=excluded.mapping",
                             (supplier, json.dumps(mapping, ensure_ascii=False)))
        return jsonify(imported=len(ids), duplicates=len(prepared)-len(ids), ids=ids)

    @app.get("/api/export.xlsx")
    def export():
        rows = [serialize(row) for row in db().execute(SUMMARY + " ORDER BY i.inspected_at DESC, i.id DESC")]
        selected_ids = request.args.get("ids")
        if selected_ids is not None:
            if any(not value.isascii() or not value.isdecimal() for value in selected_ids.split(",") if value):
                raise ValueError("엑셀 출력 대상이 올바르지 않습니다.")
            ids = {int(value) for value in selected_ids.split(",") if value}
            rows = [row for row in rows if row["id"] in ids]
        book = Workbook()
        sheet = book.active
        sheet.title = "검사 이력"
        sheet.append(["검사 ID", "협력사", "제품코드", "검사일시", "검사공정", "로트", "검사 영상", "데이터 출처",
                      "협력사 판정", "협력사 불량유형", "AI 판정", "AI 불량유형", "AI 모델 점수", "최종 판정",
                      "최종 불량유형", "담당자", "메모", "검토 시간(UTC)", "검토 차수"])
        for row in rows:
            def types(key):
                return ", ".join(CLASSES[kind] for kind in row[key] or [])

            sheet.append([row["external_id"], row["supplier"], row["product"], row["inspected_at"], row["process"],
                          row["batch"], row["image_file"], row["data_source"], row["reported_result"], row["reported_type"],
                          row["ai_result"], types("ai_types"), row["score"], row["final_result"], types("final_types"),
                          row["reviewer"], row["note"], row["reviewed_at"], row["revision"]])
            for cell in sheet[sheet.max_row]:
                if cell.data_type == "f":
                    cell.data_type = "s"
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        buffer = BytesIO()
        book.save(buffer)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="AIMPACT-inspections.xlsx")

    @app.get("/api/demo-package.zip")
    def upload_example():
        rows = json.loads((assets / "samples.json").read_text(encoding="utf-8"))[:3]
        metadata = StringIO()
        writer = csv.writer(metadata)
        writer.writerow(["관리번호", "품번", "날짜", "station", "lot_no", "judgement", "aoi_file", "golden_file"])
        buffer = BytesIO()
        with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
            filenames = set()
            for row in rows:
                writer.writerow([f"EX-{row['external_id']}", row["product"], row["inspected_at"], row["process"],
                                 row["batch"], row["reported_result"], row["image_file"], row["reference_file"]])
                filenames.update((row["image_file"], row["reference_file"]))
            for filename in sorted(filenames):
                archive.write(assets / "images" / filename, filename)
            archive.writestr("supplier-inspections.csv", metadata.getvalue().encode("utf-8-sig"))
            archive.writestr("SOURCE.txt", "DeepPCB — Tang et al. https://github.com/tangsanli5201/DeepPCB\nResearch use only.\n")
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="AIMPACT-upload-example.zip")

    return app
