#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import subprocess
import sys
import os
import time
from pathlib import Path
import gzip
import shutil
import scipy.io as io
import scipy.sparse as sp

from GMM_Demux import classify_drops
from GMM_Demux import GMM_IO

# Preserve progress in redirected logs even if the resource monitor kills us.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


# =========================
# Only change here
# =========================
DATASET = os.environ.get("GMMD_DATASET", "")

PROJECT_DIR = Path("./..")  # Adjust if this script is not in the root of the project
INPUT_DIR = Path(os.environ.get("GMMD_INPUT_DIR", PROJECT_DIR / DATASET))
OUTPUT_DIR = Path(os.environ.get("GMMD_OUTPUT_DIR", PROJECT_DIR / f"results{DATASET}" / "GMMDemux"))


def read_features(features_file):
    features_file = Path(features_file)

    if features_file.suffix == ".gz":
        features_df = pd.read_csv(
            features_file,
            sep="\t",
            header=None,
            compression="gzip",
            dtype=str
        )
    else:
        features_df = pd.read_csv(
            features_file,
            sep="\t",
            header=None,
            dtype=str
        )

    # Use column 0 as HTO names:
    # HTO_sim_01    HTO_sim_01    Antibody Capture
    hto_names = features_df.iloc[:, 0].astype(str).tolist()
    return features_df, hto_names


def read_barcodes(barcodes_file):
    barcodes_file = Path(barcodes_file)

    if barcodes_file.suffix == ".gz":
        with gzip.open(barcodes_file, "rt") as f:
            return [line.strip().split("\t")[0] for line in f if line.strip()]
    else:
        with open(barcodes_file, "r") as f:
            return [line.strip().split("\t")[0] for line in f if line.strip()]


def read_matrix(matrix_file):
    matrix_file = Path(matrix_file)

    if matrix_file.suffix == ".gz":
        with gzip.open(matrix_file, "rt") as f:
            matrix = io.mmread(f)
    else:
        matrix = io.mmread(matrix_file)

    if sp.issparse(matrix):
        matrix = matrix.tocsc()
    else:
        matrix = sp.csc_matrix(matrix)

    return matrix


def pick_file(input_dir, filename):
    input_dir = Path(input_dir)
    plain = input_dir / filename
    gz = input_dir / f"{filename}.gz"

    if gz.exists():
        return gz
    if plain.exists():
        return plain

    raise FileNotFoundError(f"Cannot find {plain} or {gz}")


def gzip_file(path, remove_original=True):
    path = Path(path)
    gz_path = Path(str(path) + ".gz")

    with open(path, "rb") as f_in:
        with gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    if remove_original:
        path.unlink()

    return gz_path


def get_random_seed():
    seed = os.environ.get("GMMD_RANDOM_SEED", "")
    if seed == "":
        return None
    return int(seed)


def run_gmmdemux_clean(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    auto_continue=True
):
    script_start_time = time.time()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== GMMDemux running script ===")
    print(f"Dataset: {DATASET}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    matrix_file = pick_file(input_dir, "matrix.mtx")
    barcodes_file = pick_file(input_dir, "barcodes.tsv")
    features_file = pick_file(input_dir, "features.tsv")

    # =========================
    # Read input
    # =========================
    print("Step 1: reading input files...")

    features_df, hto_names = read_features(features_file)
    hto_list = ",".join(hto_names)

    print(f"Detected HTO number: {len(hto_names)}")
    print(f"HTO list: {hto_list}")

    barcodes = read_barcodes(barcodes_file)
    matrix = read_matrix(matrix_file)

    print(f"Original matrix shape: {matrix.shape} (HTOs x droplets)")
    print(f"Original barcode number: {len(barcodes)}")

    if matrix.shape[1] != len(barcodes):
        raise ValueError(
            f"Matrix columns ({matrix.shape[1]}) do not match barcode number ({len(barcodes)})."
        )

    if matrix.shape[0] != len(hto_names):
        raise ValueError(
            f"Matrix rows ({matrix.shape[0]}) do not match HTO number ({len(hto_names)})."
        )

    # =========================
    # Filter empty droplets
    # =========================
    print("Step 2: filtering empty droplets...")

    cell_sums = np.asarray(matrix.sum(axis=0)).ravel()
    non_empty_mask = cell_sums > 0
    non_empty_indices = np.where(non_empty_mask)[0]

    print("Droplet HTO count summary:")
    print(f"  min: {cell_sums.min()}")
    print(f"  max: {cell_sums.max()}")
    print(f"  mean: {cell_sums.mean():.4f}")
    print(f"  first 10: {cell_sums[:10]}")

    print(f"Original droplets: {len(barcodes)}")
    print(f"Non-empty droplets: {len(non_empty_indices)}")
    print(f"Empty droplets removed: {len(barcodes) - len(non_empty_indices)}")

    if len(non_empty_indices) < 10:
        print(f"Warning: too few non-empty droplets ({len(non_empty_indices)})")
        if not auto_continue:
            response = input("Continue running? (y/n): ")
            if response.lower() != "y":
                print("Cancelled.")
                sys.exit(0)
        else:
            print("Auto continuing...")

    clean_matrix = matrix[:, non_empty_indices]
    clean_barcodes = [barcodes[i] for i in non_empty_indices]

    print(f"Filtered matrix shape: {clean_matrix.shape} (HTOs x droplets)")

    # =========================
    # Save temporary 10x files
    # =========================
    print("Step 3: saving cleaned temporary 10x files...")

    temp_dir = output_dir / "temp_clean"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    io.mmwrite(temp_dir / "matrix.mtx", clean_matrix)
    gzip_file(temp_dir / "matrix.mtx", remove_original=True)

    with open(temp_dir / "barcodes.tsv", "w") as f:
        for barcode in clean_barcodes:
            f.write(str(barcode) + "\n")
    gzip_file(temp_dir / "barcodes.tsv", remove_original=True)

    features_df.to_csv(
        temp_dir / "features.tsv",
        sep="\t",
        header=False,
        index=False
    )
    gzip_file(temp_dir / "features.tsv", remove_original=True)

    print("Temporary files:")
    for f in sorted(temp_dir.glob("*")):
        print(f"  {f.name}: {f.stat().st_size} bytes")

    # =========================
    # Run GMM-Demux
    # =========================
    print("Step 4: running GMM-Demux...")

    out_full = output_dir / "full_report"
    out_simplified = output_dir / "simplified_report"
    confidence_threshold = float(os.environ.get("GMMD_CONFIDENCE_THRESHOLD", "0.8"))
    max_full_hto = int(os.environ.get("GMMD_MAX_FULL_HTO", "20"))
    force_full = os.environ.get("GMMD_FORCE_FULL", "0") == "1"
    allow_adaptation = os.environ.get("GMMD_ALLOW_SAFE_PER_TAG", "0") == "1"

    if len(hto_names) > max_full_hto and not force_full and not allow_adaptation:
        reason = (
            f"Official GMM-Demux enumerates 2^{len(hto_names)} configurations; "
            f"the declared resource policy permits at most 2^{max_full_hto}. "
            "No approximate algorithm was substituted."
        )
        pd.DataFrame([{
            "method": "GMMDemux", "status": "resource_limit",
            "variant": "official", "n_hto": len(hto_names),
            "max_full_hto": max_full_hto, "error_message": reason,
        }]).to_csv(output_dir / "GMMDemux_runtime.csv", index=False)
        shutil.rmtree(temp_dir)
        print(reason)
        sys.exit(3)

    if len(hto_names) > max_full_hto and not force_full:
        cmd = [
            "GMMDemux-safe-per-tag",
            str(temp_dir),
            hto_list,
            "-t",
            str(confidence_threshold),
        ]
    else:
        cmd = [
            "GMM-demux",
            str(temp_dir),
            hto_list,
            "-f",
            str(out_full),
            "-s",
            str(out_simplified),
            "-o",
            str(output_dir / "SSD_mtx"),
            "-t",
            str(confidence_threshold),
            "-rs",
            str(get_random_seed() if get_random_seed() is not None else 1),
        ]

    print("Command:")
    print(" ".join(cmd))

    method_start_time = time.time()

    try:
        if cmd[0] == "GMMDemux-safe-per-tag":
            print(
                "Using safe per-tag mode because full GMM-demux enumerates all "
                f"2^{len(hto_names)} HTO combinations."
            )
            full_df, GMM_df = GMM_IO.read_cellranger(str(temp_dir), hto_names)
            high_array, low_array = classify_drops.obtain_arrays(
                GMM_df, get_random_seed()
            )
            high = np.column_stack(high_array)
            high_binary = high >= confidence_threshold
            n_high = high_binary.sum(axis=1)
            best_idx = high.argmax(axis=1)
            best_prob = high.max(axis=1)

            sample_call = np.full(GMM_df.shape[0], "Unassigned", dtype=object)
            sample_call[n_high > 1] = "Multiplet"
            singlet_mask = n_high == 1
            sample_call[singlet_mask] = np.array(hto_names, dtype=object)[best_idx[singlet_mask]]

            positive_tags = []
            for row in high_binary:
                tags = [hto_names[i] for i, val in enumerate(row) if val]
                positive_tags.append("+".join(tags) if tags else "")

            classification = pd.DataFrame({
                "droplet_id": GMM_df.index.astype(str),
                "sample_call": sample_call,
                "n_positive_tags": n_high,
                "positive_tags": positive_tags,
                "best_tag": np.array(hto_names, dtype=object)[best_idx],
                "best_high_probability": best_prob,
                "confidence_threshold": confidence_threshold,
                "mode": "safe_per_tag",
            })
            classification.to_csv(output_dir / "GMMDemux_classification.csv", index=False)
            summary_df = classification.rename(columns={"sample_call": "final_assign"})
            summary_df.to_csv(output_dir / "GMMDemux_summary.csv", index=False)
            print("Safe per-tag classification counts:")
            print(summary_df["final_assign"].value_counts(dropna=False).head(30).to_string())
            status = "completed_safe_per_tag"
            error_message = ""
        else:
            subprocess.run(
                cmd,
                check=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            print("GMM-Demux finished successfully.")

            status = "completed"
            error_message = ""

        method_end_time = time.time()
        method_runtime_sec = method_end_time - method_start_time

    except subprocess.CalledProcessError as e:
        method_end_time = time.time()
        method_runtime_sec = method_end_time - method_start_time

        print("GMM-Demux failed.")
        print(e)

        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)

        status = "failed"
        error_message = str(e.stderr) if e.stderr else str(e)

    # =========================
    # Save runtime
    # =========================
    script_end_time = time.time()
    total_runtime_sec = script_end_time - script_start_time

    runtime_df = pd.DataFrame([{
        "method": "GMMDemux",
        "dataset": DATASET,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_hto": clean_matrix.shape[0],
        "n_droplets_original": len(barcodes),
        "n_droplets_used": clean_matrix.shape[1],
        "n_empty_removed": len(barcodes) - len(non_empty_indices),
        "hto_list": hto_list,
        "command": " ".join(cmd),
        "confidence_threshold": confidence_threshold,
        "max_full_hto": max_full_hto,
        "status": status,
        "method_runtime_sec": method_runtime_sec,
        "total_runtime_sec": total_runtime_sec,
        "error_message": error_message
    }])

    runtime_df.to_csv(
        output_dir / "GMMDemux_runtime.csv",
        index=False
    )

    # =========================
    # Clean temporary files
    # =========================
    print("Step 5: cleaning temporary files...")
    shutil.rmtree(temp_dir)

    print("=== GMMDemux finished ===")
    print(f"Dataset: {DATASET}")
    print(f"Status: {status}")
    print(f"Method runtime seconds: {method_runtime_sec:.4f}")
    print(f"Total runtime seconds: {total_runtime_sec:.4f}")
    print(f"Runtime saved to: {output_dir / 'GMMDemux_runtime.csv'}")
    print(f"Result directory: {out_full}")

    if status == "failed":
        sys.exit(1)


if __name__ == "__main__":
    run_gmmdemux_clean()
