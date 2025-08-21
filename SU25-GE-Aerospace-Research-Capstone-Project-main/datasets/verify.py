#!/usr/bin/env python3
"""
Simple dataset verification script
"""

import os
from pathlib import Path


def check_visdrone():
    """Check if VisDrone dataset exists"""
    base = Path("datasets/data/visDrone")

    dirs = [
        "VisDrone2019-DET-train",
        "VisDrone2019-DET-val",
        "VisDrone2019-DET-test-dev",
    ]

    print("Checking VisDrone...")
    all_ok = True

    for d in dirs:
        path = base / d
        if path.exists():
            print(f"  PASS: {d}")
        else:
            print(f"  FAIL: {d} missing")
            all_ok = False

    return all_ok


def check_bdd100k():
    """Check if BDD100K dataset exists"""
    base = Path("datasets/data/BDD100K")

    print("Checking BDD100K...")

    if base.exists():
        print("  PASS: BDD100K folder found")
        return True
    else:
        print("  FAIL: BDD100K folder missing")
        return False


def main():
    print("=== Dataset Verification ===")
    print()

    visdrone_ok = check_visdrone()
    bdd100k_ok = check_bdd100k()

    print()
    if visdrone_ok and bdd100k_ok:
        print("SUCCESS: All datasets found!")
        print("Ready to start training.")
    else:
        print("WARNING: Some datasets missing.")
        print("Please check download instructions.")


if __name__ == "__main__":
    main()
