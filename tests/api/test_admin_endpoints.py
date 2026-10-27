import secrets

import pytest

from app.core.errors import OllamaUnavailable
from tests.api.conftest import TOKEN, Env
from tests.helpers import make_pdf, words


# --- authentication ------------------------------------------------------------------


@pytest.mark.parametrize("method,url,kwargs", [
    ("post", "/documents/ingest", {"json": {"path": "a.pdf"}}),
    ("delete", "/documents/pdf-x", {"params": {"confirm": "true"}}),
])
def test_admin_endpoints_need_a_token(env, method, url, kwargs):
    r = getattr(env.client, method)(url, **kwargs)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("header", ["Bearer wrong-token-wrong-token", "Bearer ", "Basic abc", "Bearer", TOKEN,
                                    f"Bearer {TOKEN} extra", f"Bearer {TOKEN[:-1]}", f"Bearer {TOKEN}x"])
def test_wrong_or_malformed_credentials_are_rejected(env, header):
    r = env.client.post("/documents/ingest", json={"path": "a.pdf"}, headers={"Authorization": header})
    assert r.status_code in (401, 403)
    assert TOKEN not in r.text


def test_wrong_token_is_403_and_missing_is_401(env):
    assert env.client.post("/documents/ingest", json={"path": "a.pdf"},
                           headers={"Authorization": "Bearer " + "z" * 30}).status_code == 403
    assert env.client.post("/documents/ingest", json={"path": "a.pdf"}).status_code == 401


def test_authentication_happens_before_body_validation(env):
    assert env.client.post("/documents/ingest", json={"nonsense": 1}).status_code == 401
    assert env.client.post("/documents/ingest", content=b"not json",
                           headers={"content-type": "application/json"}).status_code == 401


def test_endpoints_are_disabled_when_no_token_is_configured(tmp_path):
    e = Env(tmp_path, token=None)
    try:
        e.add_pdf()
        for headers in ({}, {"Authorization": "Bearer anything-at-all-123456"}):
            r = e.client.post("/documents/ingest", json={"path": "terraform.pdf"}, headers=headers)
            assert r.status_code == 403 and "disabled" in r.json()["detail"]
            assert e.client.delete("/documents/x", params={"confirm": "true"}, headers=headers).status_code == 403
        assert e.client.get("/documents").json()["count"] == 0
    finally:
        e.close()


def test_the_token_is_compared_in_constant_time(env, monkeypatch):
    calls = []
    real = secrets.compare_digest
    monkeypatch.setattr("app.api.security.secrets.compare_digest", lambda a, b: calls.append((a, b)) or real(a, b))
    env.client.post("/documents/ingest", json={"path": "a.pdf"}, headers={"Authorization": "Bearer nope-nope-nope-nope"})
    assert len(calls) == 1 and all(isinstance(x, bytes) for x in calls[0])


def test_read_endpoints_do_not_require_a_token(loaded):
    for url in ("/health", "/documents"):
        assert loaded.client.get(url).status_code == 200
    assert loaded.client.post("/search", json={"query": "q"}).status_code == 200


# --- ingest ------------------------------------------------------------------------


def test_ingest_a_file_relative_to_the_documents_directory(env):
    env.add_pdf()
    r = env.ingest("terraform.pdf", category="iac-automation", topics=["Terraform"])
    assert r.status_code == 200
    body = r.json()
    assert body["counts"] == {"indexed": 1} and body["results"][0]["path"] == "terraform.pdf"
    doc = env.client.get("/documents").json()["documents"][0]
    assert (doc["category"], doc["topics"], doc["status"]) == ("iac-automation", ["terraform"], "indexed")
    again = env.ingest("terraform.pdf").json()
    assert again["counts"] == {"unchanged": 1}


def test_ingest_accepts_an_absolute_path_inside_the_directory(env):
    pdf = env.add_pdf()
    assert env.ingest(str(pdf)).json()["counts"] == {"indexed": 1}


def test_ingest_a_folder_is_recursive_and_reports_relative_paths(env):
    env.add_pdf("a.pdf", "terraform")
    make_pdf(env.docs / "nested" / "b.pdf", [words("bt", 60)])
    r = env.ingest(".", category="general")
    body = r.json()
    assert r.status_code == 200 and body["counts"] == {"indexed": 2}
    assert sorted(x["path"] for x in body["results"]) == ["a.pdf", "nested/b.pdf"]
    assert str(env.root) not in r.text


def test_force_reindex_and_invalid_slugs(env):
    env.add_pdf()
    env.ingest("terraform.pdf")
    assert env.ingest("terraform.pdf", force=True).json()["counts"] == {"reindexed": 1}
    assert env.ingest("terraform.pdf", category="Bad Slug!").status_code == 422
    assert env.ingest("terraform.pdf", topics=["../x"]).status_code == 422


# --- path confinement ---------------------------------------------------------------


@pytest.mark.parametrize("path", ["../outside.pdf", "../../etc/passwd", "nested/../../outside.pdf", "/etc/passwd", "/"])
def test_paths_outside_the_documents_directory_are_forbidden(env, path):
    make_pdf(env.root / "outside.pdf", [words("ot", 40)])
    r = env.ingest(path)
    assert r.status_code == 403 and "outside" in r.json()["detail"]
    assert env.client.get("/documents").json()["count"] == 0


def test_a_symlink_pointing_outside_is_forbidden(env):
    outside = make_pdf(env.root / "outside.pdf", [words("ot", 40)])
    (env.docs / "link.pdf").symlink_to(outside)
    assert env.ingest("link.pdf").status_code == 403
    (env.docs / "dirlink").symlink_to(env.root)
    assert env.ingest("dirlink").status_code == 403
    assert env.client.get("/documents").json()["count"] == 0


def test_symlinked_files_inside_a_folder_are_skipped_when_they_leave_the_directory(env):
    outside = make_pdf(env.root / "outside.pdf", [words("ot", 40)], title="Outside Book")
    env.add_pdf("inside.pdf")
    (env.docs / "link.pdf").symlink_to(outside)
    body = env.ingest(".").json()
    assert body["counts"] == {"indexed": 1}
    assert {d["title"] for d in env.client.get("/documents").json()["documents"]} == {"Terraform Handbook"}


def test_missing_files_are_a_404_without_leaking_paths(env):
    r = env.ingest("nope.pdf")
    assert r.status_code == 404 and str(env.root) not in r.text


def test_problem_files_are_reported_per_file_not_as_a_server_error(env):
    (env.docs / "broken.pdf").write_bytes(b"not a pdf")
    (env.docs / "notes.txt").write_text("plain text")
    one = env.ingest("broken.pdf")
    assert one.status_code == 200 and one.json()["results"][0]["outcome"] == "error"
    txt = env.ingest("notes.txt").json()["results"][0]
    assert txt["outcome"] == "error" and "unsupported" in txt["message"]


def test_embedding_service_down_during_ingest_is_a_503(env):
    env.add_pdf()
    env.embedder.error = OllamaUnavailable("cannot reach Ollama at http://127.0.0.1:11434")
    r = env.ingest("terraform.pdf")
    assert r.status_code == 503 and "127.0.0.1" not in r.text
    assert env.services.vectors.count() == 0


def test_only_one_ingestion_runs_at_a_time(env):
    env.add_pdf()
    assert env.services.ingest_lock.acquire(blocking=False)
    try:
        r = env.ingest("terraform.pdf")
        assert r.status_code == 409 and "already running" in r.json()["detail"]
    finally:
        env.services.ingest_lock.release()
    assert env.ingest("terraform.pdf").status_code == 200, "the lock is released afterwards"


def test_the_lock_is_released_even_when_ingestion_fails(env):
    env.add_pdf()
    env.embedder.error = OllamaUnavailable("down")
    assert env.ingest("terraform.pdf").status_code == 503
    env.embedder.error = None
    assert env.ingest("terraform.pdf").status_code == 200


# --- delete ------------------------------------------------------------------------


def test_delete_requires_explicit_confirmation(loaded):
    doc_id = loaded.client.get("/documents", params={"category": "iac-automation"}).json()["documents"][0]["document_id"]
    for params in ({}, {"confirm": "false"}):
        r = loaded.client.delete(f"/documents/{doc_id}", params=params, headers=loaded.auth)
        assert r.status_code == 400 and "confirm=true" in r.json()["detail"]
    assert loaded.client.get(f"/documents/{doc_id}").status_code == 200


def test_delete_removes_the_index_entry_but_keeps_the_file(loaded):
    doc_id = loaded.client.get("/documents", params={"category": "iac-automation"}).json()["documents"][0]["document_id"]
    chunks_before = loaded.services.vectors.count()
    r = loaded.client.delete(f"/documents/{doc_id}", params={"confirm": "true"}, headers=loaded.auth)
    assert r.status_code == 200 and r.json() == {"document_id": doc_id, "removed": True}
    assert loaded.client.get(f"/documents/{doc_id}").status_code == 404
    assert loaded.services.vectors.count_document(doc_id) == 0 and loaded.services.vectors.count() < chunks_before
    assert (loaded.docs / "terraform.pdf").is_file(), "the original file must never be deleted"
    assert loaded.client.get("/documents").json()["count"] == 1
    assert loaded.client.delete(f"/documents/{doc_id}", params={"confirm": "true"}, headers=loaded.auth).status_code == 404


def test_delete_is_refused_while_an_ingestion_runs(loaded):
    doc_id = loaded.client.get("/documents").json()["documents"][0]["document_id"]
    assert loaded.services.ingest_lock.acquire(blocking=False)
    try:
        r = loaded.client.delete(f"/documents/{doc_id}", params={"confirm": "true"}, headers=loaded.auth)
        assert r.status_code == 409
    finally:
        loaded.services.ingest_lock.release()
    assert loaded.client.get(f"/documents/{doc_id}").status_code == 200
