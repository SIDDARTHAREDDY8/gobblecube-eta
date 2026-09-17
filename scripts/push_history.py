#!/usr/bin/env python3
"""Create the public submission repo and replay local git history onto it.

Why replay instead of gh_push.py's single commit: Gobblecube reads the git
log as a hiring signal ("first commits rarely look like final ones"), so the
real trajectory (skeleton -> cleaning -> CRS fix -> sample validation ->
full-scale training) is part of the submission.

Steps:
  1. Create public repo SIDDARTHAREDDY8/<name> (or reuse if it exists).
  2. Seed it with one contents-API PUT (brand-new empty repos 409 on the
     Git Data API blobs endpoint; the seed avoids that).
  3. For each local commit, oldest -> newest: upload any new blobs, build a
     tree on top of the previous tree, create a commit preserving message,
     author, and date, and fast-forward refs/heads/main.
  4. Verify: ref SHA == last commit SHA, and the recursive tree contains
     predict.py, Dockerfile, artifacts/model.pkl, artifacts/lookups.npz,
     README.md.

Usage:
    python scripts/push_history.py --repo gobblecube-eta [--dry-run]

Requires the files to be committed locally first (including force-added
artifacts/*.pkl / *.npz, which .gitignore excludes by default).
"""
from __future__ import annotations

import argparse
import base64
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path.home() / "workspace" / "skills" / "github" / "bin"))
from gh_api import call  # noqa: E402

OWNER = "SIDDARTHAREDDY8"
REQUIRED_PATHS = {"predict.py", "Dockerfile", "artifacts/model.pkl",
                  "artifacts/lookups.npz", "README.md"}


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=HERE, capture_output=True, text=True,
                          check=True).stdout


def api(method: str, path: str, data=None):
    st, out = call(method, path, data)
    if st not in (200, 201):
        raise SystemExit(f"GitHub API {method} {path} -> {st}: {out}")
    return out


def ensure_repo(name: str) -> None:
    st, out = call("GET", f"/repos/{OWNER}/{name}")
    if st == 200:
        print(f"repo exists: {out['html_url']}")
        return
    repo = api("POST", "/user/repos", {
        "name": name, "private": False, "auto_init": False,
        "description": "NYC taxi trip-duration prediction — Gobblecube AI "
                       "Builders hiring-challenge submission (ETA track).",
    })
    print(f"created repo: {repo['html_url']}")
    # Seed: empty repos 409 on the Git Data API blobs endpoint.
    seed = api("PUT", f"/repos/{OWNER}/{name}/contents/README.md", {
        "message": "seed", "content": base64.b64encode(b"# seed\n").decode(),
    })
    print(f"seeded ({seed['commit']['sha'][:12]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="gobblecube-eta")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    commits = sh("git", "log", "--format=%H|%an|%ae|%aI|%s", "--reverse",
                 "master").strip().split("\n")
    print(f"{len(commits)} local commits to replay")
    if args.dry_run:
        for c in commits:
            print("  ", c.split("|")[-1])
        return

    ensure_repo(args.repo)

    ref = api("GET",
              f"/repos/{OWNER}/{args.repo}/git/matching-refs/heads/")
    # The seed commit created the default branch; use whatever it is.
    parent = None
    branch = None
    for r in ref:
        if r["ref"] == "refs/heads/main":
            parent = r["object"]["sha"]
            branch = "main"
            break
    if parent is None and ref:
        branch = ref[0]["ref"].split("/")[-1]
        parent = ref[0]["object"]["sha"]
        print(f"note: default branch is {branch}, not main")
    if parent is None:
        raise SystemExit("no branches after seeding?!"
                         f" refs seen: {[r['ref'] for r in ref]}")
    print(f"seed parent on {branch}: {parent[:12]}")

    blob_cache: dict[str, str] = {}  # local blob sha -> github blob sha
    for line in commits:
        sha, an, ae, ad, subject = line.split("|", 4)
        # unique blobs in this commit's tree
        entries = []
        for t in sh("git", "ls-tree", "-r", "-z", sha).split("\0"):
            if not t:
                continue
            meta, path = t.split("\t")
            mode, typ, bsha = meta.split()
            if typ != "blob":
                print(f"  skip non-blob {path} ({typ})", flush=True)
                continue
            if bsha not in blob_cache:
                content = subprocess.run(
                    ["git", "cat-file", "-p", bsha], cwd=HERE,
                    capture_output=True, check=True).stdout
                mb = len(content) / 1e6
                print(f"  blob {path} ({mb:.1f} MB) ...", flush=True)
                b = api("POST", f"/repos/{OWNER}/{args.repo}/git/blobs",
                        {"content": base64.b64encode(content).decode(),
                         "encoding": "base64"})
                blob_cache[bsha] = b["sha"]
            entries.append({"path": path, "mode": mode, "type": "blob",
                            "sha": blob_cache[bsha]})
        tree = api("POST", f"/repos/{OWNER}/{args.repo}/git/trees",
                   {"tree": entries})  # full tree each commit; no base_tree
        commit = api("POST", f"/repos/{OWNER}/{args.repo}/git/commits", {
            "message": sh("git", "log", "--format=%B", "-n", "1",
                          sha).strip() or subject,
            "tree": tree["sha"],
            "parents": [parent],
            "author": {"name": an, "email": ae, "date": ad},
            "committer": {"name": an, "email": ae, "date": ad},
        })
        parent = commit["sha"]
        print(f"replayed {sha[:12]} -> {parent[:12]}  {subject[:60]}")

    api("PATCH", f"/repos/{OWNER}/{args.repo}/git/refs/heads/{branch}",
        {"sha": parent})
    print(f"refs/heads/{branch} -> {parent[:12]}")

    # ---- verify (never trust a push claim) ----
    r = api("GET", f"/repos/{OWNER}/{args.repo}/git/ref/heads/{branch}")
    assert r["object"]["sha"] == parent, "ref SHA mismatch!"
    commit_obj = api("GET",
                     f"/repos/{OWNER}/{args.repo}/git/commits/{parent}")
    tree_sha = commit_obj["tree"]["sha"]
    t = api("GET",
            f"/repos/{OWNER}/{args.repo}/git/trees/{tree_sha}?recursive=1")
    assert not t.get("truncated"), "tree was truncated!"
    paths = {e["path"] for e in t["tree"] if e["type"] == "blob"}
    missing = REQUIRED_PATHS - paths
    assert not missing, f"missing from pushed tree: {missing}"
    big = [(e["path"], e["size"]) for e in t["tree"]
           if e["type"] == "blob" and e["path"] in REQUIRED_PATHS]
    print("verified on GitHub:")
    for p, s in sorted(big):
        print(f"  {p} ({s/1e6:.1f} MB)")
    print(f"https://github.com/{OWNER}/{args.repo}  @ {parent[:12]}")


if __name__ == "__main__":
    main()
