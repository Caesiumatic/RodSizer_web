from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
import re
import time
import asyncio
from pathlib import Path
from typing import List
from processing import process_image, generate_preview, save_results_to_excel, generate_binary_mask_preview
from pydantic import BaseModel
import json
from contextlib import asynccontextmanager

SESSION_TTL_SECONDS = 24 * 60 * 60  # 24 hours
CLEANUP_INTERVAL_SECONDS = 60 * 60  # run every 1 hour

# Upload limits (Hugging Face Spaces free tier has limited ephemeral storage).
MAX_UPLOAD_BYTES = 200 * 1024 * 1024   # 200 MB per file
MAX_UPLOAD_FILES = 100                  # per request

# Strict input patterns — these flow into filesystem paths, so we must not accept
# separators, traversal tokens, or glob metacharacters.
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
# Image IDs are the stem of uploaded files, formatted as "<uuid>" or "<uuid>_<sanitized_name>".
_IMAGE_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?:_.+)?$"
)
_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")

# Roots for session storage. Resolving at import time lets us reject any path
# that escapes them regardless of how components were joined.
_UPLOAD_ROOT = Path("/tmp/uploads").resolve()
_RESULTS_ROOT = Path("/tmp/results").resolve()

def cleanup_old_sessions():
    """Delete session dirs in /tmp/uploads and /tmp/results not modified in >24h."""
    now = time.time()
    for base in [Path("/tmp/uploads"), Path("/tmp/results")]:
        if not base.exists():
            continue
        for session_dir in base.iterdir():
            if not session_dir.is_dir():
                continue
            age = now - session_dir.stat().st_mtime
            if age > SESSION_TTL_SECONDS:
                shutil.rmtree(session_dir, ignore_errors=True)
                print(f"[cleanup] Removed old session: {session_dir}")

async def periodic_cleanup():
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        cleanup_old_sessions()

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_old_sessions()  # clean on startup
    task = asyncio.create_task(periodic_cleanup())
    yield
    task.cancel()

class ExportRequest(BaseModel):
    image_id: str
    selected_ids: List[int]

app = FastAPI(title="RodSizer", lifespan=lifespan)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base directories (session subdirs are created on demand)
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
RESULTS_SCHEMA_VERSION = 2

FRONTEND_DIR.mkdir(exist_ok=True)


def _validate_session_id(session_id: str) -> str:
    """Reject anything that could escape the per-session directory.
    Session IDs come from clients (JS generates them) so we must treat them as untrusted."""
    if not session_id or not _SESSION_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    return session_id


def _safe_session_dir(root: Path, session_id: str) -> Path:
    _validate_session_id(session_id)
    candidate = (root / session_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session id")
    return candidate


def get_upload_dir(session_id: str) -> Path:
    d = _safe_session_dir(_UPLOAD_ROOT, session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_results_dir(session_id: str) -> Path:
    d = _safe_session_dir(_RESULTS_ROOT, session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_join(base: Path, *parts: str) -> Path:
    """Join under `base` and ensure the result stays inside `base` after resolving.
    Raises HTTPException(400) on traversal attempts or empty components."""
    base_resolved = base.resolve()
    candidate = base_resolved
    for raw in parts:
        if raw is None or raw == "":
            raise HTTPException(status_code=400, detail="Invalid path component")
        if "/" in raw or "\\" in raw or raw in (".", ".."):
            raise HTTPException(status_code=400, detail="Invalid path component")
        candidate = candidate / raw

    resolved = candidate.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path escapes allowed directory")
    return resolved


def _validate_image_id(image_id: str) -> str:
    if not image_id or not _IMAGE_ID_RE.match(image_id):
        raise HTTPException(status_code=400, detail="Invalid image id")
    for c in image_id:
        if ord(c) < 32 or c in '/\\*?[]':
            raise HTTPException(status_code=400, detail="Invalid image id")
    return image_id


def _resolve_folder(upload_dir: Path, folder_name: str) -> Path:
    safe = _sanitize_folder_name(folder_name)
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid folder name")
    return _safe_join(upload_dir, safe)


def _cached_results_are_current(payload: dict, expected_binary_mask_tune: int = 0) -> bool:
    if payload.get("results_schema_version") != RESULTS_SCHEMA_VERSION:
        return False
    if int(payload.get("binary_mask_tune", 0)) != int(expected_binary_mask_tune):
        return False

    calibration_info = payload.get("calibration_info") or {}
    method = calibration_info.get("method")

    if method == "default":
        return False
    if method == "uncalibrated" and "is_placeholder" not in calibration_info:
        return False

    return True


def _sanitize_folder_name(folder_name: str) -> str:
    if folder_name is None:
        return ""

    invalid_chars = '<>:"/\\|?*'
    cleaned = "".join(
        c for c in folder_name
        if ord(c) >= 32 and c not in invalid_chars
    )

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.rstrip(". ")
    if cleaned in ("", ".", ".."):
        return ""
    return cleaned


def _find_input_and_calibration_source(image_id: str, upload_dir: Path):
    _validate_image_id(image_id)
    files = [p for p in upload_dir.rglob("*") if p.is_file() and p.stem == image_id]
    if not files:
        raise HTTPException(status_code=404, detail="Image not found")

    input_path = files[0]
    search_dir = input_path.parent
    calibration_source_path = None

    current_filename = input_path.name
    if len(current_filename) > 37 and current_filename[36] == '_':
        original_name_with_ext = current_filename[37:]
        original_stem = Path(original_name_with_ext).stem
    else:
        original_stem = input_path.stem

    if original_stem:
        for f in search_dir.glob("*"):
            if f.suffix.lower() in ['.dm3', '.dm4', '.emd']:
                if len(f.name) > 37 and f.name[36] == '_':
                    cal_stem = Path(f.name[37:]).stem
                else:
                    cal_stem = f.stem

                if cal_stem == original_stem:
                    calibration_source_path = f
                    break

    return input_path, calibration_source_path

# Serve Frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def read_index():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/analysis")
async def read_analysis():
    return FileResponse(FRONTEND_DIR / "analysis.html")

@app.get("/folder_analysis")
async def read_folder_analysis():
    return FileResponse(FRONTEND_DIR / "folder_analysis.html")

# --- Folder Management ---

@app.post("/folders")
async def create_folder(folder_name: str = Form(...), session_id: str = Query(...)):
    try:
        upload_dir = get_upload_dir(session_id)
        safe_name = _sanitize_folder_name(folder_name)
        if not safe_name:
             raise HTTPException(status_code=400, detail="Invalid folder name")

        folder_path = _safe_join(upload_dir, safe_name)
        if folder_path.exists():
             raise HTTPException(status_code=400, detail="Folder already exists")

        folder_path.mkdir(parents=True, exist_ok=True)
        return {"status": "success", "folder": safe_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/folders")
async def list_folders(session_id: str = Query(...)):
    upload_dir = get_upload_dir(session_id)
    folders = []
    for path in upload_dir.iterdir():
        if path.is_dir():
            folders.append(path.name)
    folders.sort()
    return folders


@app.put("/folders/{folder_name}")
async def rename_folder(folder_name: str, new_name: str = Form(...), session_id: str = Query(...)):
    try:
        upload_dir = get_upload_dir(session_id)
        source_path = _resolve_folder(upload_dir, folder_name)
        if not source_path.exists() or not source_path.is_dir():
            raise HTTPException(status_code=404, detail="Folder not found")

        safe_name = _sanitize_folder_name(new_name)
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid folder name")

        if safe_name == folder_name:
            return {"status": "success", "folder": safe_name, "renamed": False}

        target_path = _safe_join(upload_dir, safe_name)
        if target_path.exists():
            raise HTTPException(status_code=400, detail="A folder with that name already exists")

        source_path.rename(target_path)
        return {"status": "success", "folder": safe_name, "renamed": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/folders/{folder_name}")
async def delete_folder(folder_name: str, session_id: str = Query(...)):
    try:
        upload_dir = get_upload_dir(session_id)
        folder_path = _resolve_folder(upload_dir, folder_name)
        if not folder_path.exists() or not folder_path.is_dir():
            raise HTTPException(status_code=404, detail="Folder not found")

        shutil.rmtree(folder_path)
        return {"status": "success", "message": "Folder deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Folder Aggregation ---

class FolderSelectionRequest(BaseModel):
    image_id: str
    selected_ids: List[int]

@app.post("/folders/{folder_name}/save_selection")
async def save_folder_selection(folder_name: str, req: FolderSelectionRequest, session_id: str = Query(...)):
    try:
        _validate_image_id(req.image_id)
        upload_dir = get_upload_dir(session_id)
        results_dir = get_results_dir(session_id)
        folder_path = _resolve_folder(upload_dir, folder_name)
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="Folder not found")

        cache_dir = folder_path / ".analysis_cache"
        cache_dir.mkdir(exist_ok=True)

        json_path = results_dir / f"{req.image_id}_results.json"
        if not json_path.exists():
             raise HTTPException(status_code=404, detail="Original analysis not found")

        with open(json_path) as f:
            data = json.load(f)

        full_results = data.get("data", [])
        filtered = [r for r in full_results if r["id"] in req.selected_ids]

        if not filtered:
             raise HTTPException(status_code=400, detail="No particles selected to save")

        save_payload = {
            "image_id": req.image_id,
            "filename": data.get("filename", req.image_id),
            "data": filtered,
            "timestamp": data.get("timestamp", ""),
            "pixel_size_nm": data.get("pixel_size_nm", 0)
        }

        output_path = cache_dir / f"{req.image_id}.json"
        with open(output_path, "w") as f:
            json.dump(save_payload, f, indent=2)

        return {"status": "success", "count": len(filtered)}

    except Exception as e:
        print(f"Save Selection Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/folders/{folder_name}/aggregate")
async def aggregate_folder(folder_name: str, session_id: str = Query(...)):
    try:
        upload_dir = get_upload_dir(session_id)
        folder_path = _resolve_folder(upload_dir, folder_name)
        cache_dir = folder_path / ".analysis_cache"

        if not cache_dir.exists():
            return {"data": [], "stats": {}, "file_count": 0}

        combined_data = []
        files = list(cache_dir.glob("*.json"))

        for p in files:
            with open(p) as f:
                payload = json.load(f)
                fname = payload.get("filename", "unknown")
                for p_data in payload.get("data", []):
                    p_data["source_image"] = fname
                    combined_data.append(p_data)

        stats = {}
        if combined_data:
            import pandas as pd
            df = pd.DataFrame(combined_data)

            def get_stat(col):
                if col not in df: return 0
                return round(float(df[col].mean()), 1), round(float(df[col].std()), 1)

            l_m, l_s = get_stat("length_nm")
            w_m, w_s = get_stat("width_nm")
            ar_m, ar_s = get_stat("aspect_ratio")

            stats = {
                "count": len(combined_data),
                "mean_length": f"{l_m} ± {l_s}",
                "mean_width": f"{w_m} ± {w_s}",
                "mean_ar": f"{ar_m} ± {ar_s}"
            }

        return {
            "data": combined_data,
            "stats": stats,
            "file_count": len(files)
        }

    except Exception as e:
        print(f"Aggregate Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/folders/{folder_name}/export_aggregate")
async def export_aggregate_folder(folder_name: str, session_id: str = Query(...)):
    try:
        upload_dir = get_upload_dir(session_id)
        results_dir = get_results_dir(session_id)
        folder_path = _resolve_folder(upload_dir, folder_name)
        safe_folder_name = folder_path.name
        cache_dir = folder_path / ".analysis_cache"

        if not cache_dir.exists():
            raise HTTPException(status_code=404, detail="No data to export")

        combined_data = []
        files = list(cache_dir.glob("*.json"))

        for p in files:
            with open(p) as f:
                payload = json.load(f)
                fname = payload.get("filename", "unknown")
                for p_data in payload.get("data", []):
                    p_data["source_image"] = fname
                    combined_data.append(p_data)

        if not combined_data:
            raise HTTPException(status_code=400, detail="No data found")

        temp_name = f"export_folder_{safe_folder_name}_{uuid.uuid4().hex[:8]}.xlsx"
        temp_path = results_dir / temp_name

        save_results_to_excel(combined_data, temp_path)

        return FileResponse(
            path=temp_path,
            filename=f"{safe_folder_name}_analysis_report.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Export Aggregate Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------

@app.post("/upload")
async def upload_images(background_tasks: BackgroundTasks, folder: str = Form(None), files: List[UploadFile] = File(...), session_id: str = Query(...)):
    upload_dir = get_upload_dir(session_id)
    results_dir = get_results_dir(session_id)
    uploaded_files = []
    try:
        if len(files) > MAX_UPLOAD_FILES:
            raise HTTPException(status_code=400, detail=f"Too many files (max {MAX_UPLOAD_FILES} per request)")

        if folder:
            save_dir = _resolve_folder(upload_dir, folder)
            if not save_dir.exists():
                raise HTTPException(status_code=400, detail="Folder does not exist")
        else:
            save_dir = upload_dir

        for file in files:
            file_id = str(uuid.uuid4())
            raw_name = Path(file.filename or "").name
            if not raw_name or raw_name in (".", ".."):
                raise HTTPException(status_code=400, detail="Invalid filename")
            safe_filename = _sanitize_folder_name(raw_name)
            if not safe_filename:
                raise HTTPException(status_code=400, detail="Invalid filename")
            save_name = f"{file_id}_{safe_filename}"

            file_path = _safe_join(save_dir, save_name)

            bytes_written = 0
            with open(file_path, "wb") as buffer:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > MAX_UPLOAD_BYTES:
                        buffer.close()
                        try:
                            file_path.unlink()
                        except OSError:
                            pass
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                        )
                    buffer.write(chunk)

            generate_preview(file_path, results_dir)

            uploaded_files.append({
                "id": file_path.stem,
                "filename": safe_filename,
                "status": "processing"
            })

            search_dir = file_path.parent
            calibration_source_path = None
            original_stem = None

            if len(save_name) > 37 and save_name[36] == '_':
                original_stem = Path(save_name[37:]).stem
            else:
                original_stem = Path(save_name).stem

            for f in search_dir.glob("*"):
                if f.suffix.lower() in ['.dm3', '.dm4', '.emd']:
                    dm3_stem = None
                    if len(f.name) > 37 and f.name[36] == '_':
                        dm3_stem = Path(f.name[37:]).stem
                    else:
                        dm3_stem = f.stem
                    if dm3_stem == original_stem:
                        calibration_source_path = f
                        break

            background_tasks.add_task(process_image, file_path, results_dir, None, calibration_source_path)

        return uploaded_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images")
async def list_images(session_id: str = Query(...), folder: str = Query(None)):
    upload_dir = get_upload_dir(session_id)
    results_dir = get_results_dir(session_id)
    images = []

    if folder:
        try:
            target_dir = _resolve_folder(upload_dir, folder)
        except HTTPException:
            return []
        if not target_dir.exists():
            return []
    else:
        target_dir = upload_dir

    for path in target_dir.glob("*"):
        if path.is_file() and path.suffix.lower() in ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.dm3', '.dm4', '.emd']:
            display_name = path.name
            if len(path.name) > 37 and path.name[36] == '_':
                display_name = path.name[37:]

            image_id = path.stem
            overlay_path = results_dir / f"{image_id}_overlay.jpg"
            status = "complete" if overlay_path.exists() else "processing"

            images.append({
                "id": image_id,
                "filename": path.name,
                "display_name": display_name,
                "status": status
            })
    images.sort(key=lambda x: x['display_name'])
    return images

@app.delete("/images/{image_id}")
async def delete_image(image_id: str, session_id: str = Query(...)):
    try:
        _validate_image_id(image_id)
        upload_dir = get_upload_dir(session_id)
        results_dir = get_results_dir(session_id)
        files = [p for p in upload_dir.rglob("*") if p.is_file() and p.stem == image_id]
        if not files:
            raise HTTPException(status_code=404, detail="Image not found")

        for f in files:
            resolved = f.resolve()
            try:
                resolved.relative_to(upload_dir)
            except ValueError:
                continue
            os.remove(resolved)

        for res_file in results_dir.iterdir():
            if not res_file.is_file() or not res_file.name.startswith(image_id):
                continue
            resolved = res_file.resolve()
            try:
                resolved.relative_to(results_dir)
            except ValueError:
                continue
            os.remove(resolved)

        return {"status": "success", "message": "Image deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/{image_id}")
async def process_image_endpoint(
    image_id: str,
    session_id: str = Query(...),
    manual_pixel_size: float = None,
    requested_bar_length_nm: float = None,
    force_reprocess: bool = False,
    binary_mask_tune: int = 0
):
    try:
        upload_dir = get_upload_dir(session_id)
        results_dir = get_results_dir(session_id)
        input_path, calibration_source_path = _find_input_and_calibration_source(image_id, upload_dir)

        if not manual_pixel_size and not force_reprocess:
            results_path = results_dir / f"{image_id}_results.json"
            if results_path.exists():
                import json
                try:
                    with open(results_path, 'r') as f:
                        cached = json.load(f)
                    if _cached_results_are_current(cached, expected_binary_mask_tune=binary_mask_tune):
                        return cached
                except Exception:
                    pass

        result = process_image(
            input_path,
            results_dir,
            manual_pixel_size,
            calibration_source_path,
            requested_bar_length_nm,
            binary_mask_tune
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/{image_id}/binary_preview")
async def process_binary_preview_endpoint(
    image_id: str,
    session_id: str = Query(...),
    manual_pixel_size: float = None,
    binary_mask_tune: int = 0
):
    try:
        upload_dir = get_upload_dir(session_id)
        results_dir = get_results_dir(session_id)
        input_path, calibration_source_path = _find_input_and_calibration_source(image_id, upload_dir)
        return generate_binary_mask_preview(
            input_path,
            results_dir,
            manual_pixel_size,
            calibration_source_path,
            binary_mask_tune
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{filename}")
async def get_result_file(filename: str, session_id: str = Query(...)):
    if (
        not filename
        or filename in (".", "..")
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or any(ord(c) < 32 for c in filename)
        or any(ch in filename for ch in "*?[]")
    ):
        raise HTTPException(status_code=400, detail="Invalid filename")

    upload_dir = get_upload_dir(session_id)
    results_dir = get_results_dir(session_id)

    results_candidate = (results_dir / filename).resolve()
    try:
        results_candidate.relative_to(results_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if results_candidate.is_file():
        return FileResponse(results_candidate)

    uploads_candidate = (upload_dir / filename).resolve()
    try:
        uploads_candidate.relative_to(upload_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if uploads_candidate.is_file():
        return FileResponse(uploads_candidate)

    for found in upload_dir.rglob(filename):
        resolved = found.resolve()
        try:
            resolved.relative_to(upload_dir)
        except ValueError:
            continue
        if resolved.is_file() and resolved.name == filename:
            return FileResponse(resolved)

    raise HTTPException(status_code=404, detail="File not found")

# --- Export ---

@app.post("/export")
async def export_data(req: ExportRequest, session_id: str = Query(...)):
    try:
        _validate_image_id(req.image_id)
        results_dir = get_results_dir(session_id)
        json_path = results_dir / f"{req.image_id}_results.json"
        if not json_path.exists():
            raise HTTPException(status_code=404, detail="Results not found")

        with open(json_path) as f:
            data = json.load(f)

        full_results = data.get("data", [])
        filtered_results = [r for r in full_results if r["id"] in req.selected_ids]

        if not filtered_results:
             raise HTTPException(status_code=400, detail="No particles selected")

        temp_name = f"export_{req.image_id}_{uuid.uuid4().hex[:8]}.xlsx"
        temp_path = results_dir / temp_name

        save_results_to_excel(filtered_results, temp_path)

        # Sanitize the suggested download filename — it originates from the
        # uploaded filename and flows into a response header.
        raw_name = data.get("filename") or req.image_id
        safe_download_stem = _sanitize_folder_name(Path(raw_name).stem) or req.image_id

        return FileResponse(
            path=temp_path,
            filename=f"{safe_download_stem}_detected.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Export Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
