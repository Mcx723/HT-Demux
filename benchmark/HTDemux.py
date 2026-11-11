#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from classifier import HTClassifier
from preprocess import DemuxManager


def main():
    parser = argparse.ArgumentParser(description="Run HT-Demux on a 10x-style HTO matrix.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", default="auto", choices=["auto", "NB", "GMM", "DLN"],
                        help="Auto retains enumerated NB when affordable; implicit inference selects NB/DLN by unlabeled count-likelihood BIC")
    parser.add_argument("--method", default="EM", choices=["EM", "GD"])
    parser.add_argument("--max-klet", type=int, default=None,
                        help="Maximum number of distinct positive tags; default auto adds orders 3/4 when affordable")
    parser.add_argument("--max-auto-configs", type=int, default=4096,
                        help="Budget for adding orders 3/4 in auto mode; singles/pairs are always retained")
    parser.add_argument("--map-thresh", type=float, default=0.9)
    parser.add_argument("--decision-mode", default="hierarchical", choices=["hierarchical", "joint", "posterior", "map", "marginal"],
                        help="hierarchical compares total singlet/multiplet/negative posterior; only singlets receive a tag identity")
    parser.add_argument("--marginal-singlet-thresh", type=float, default=0.05,
                        help="Only used with --decision-mode marginal")
    parser.add_argument("--marginal-doublet-thresh", type=float, default=0.99,
                        help="Only used with --decision-mode marginal")
    parser.add_argument("--posterior-thresh", type=float, default=0.5,
                        help="Legacy cutoff for posterior decisions; hierarchical mode uses minimum expected 0-1 loss and ignores this cutoff")
    parser.add_argument("--multiplet-prior-scale", type=float, default=0.5,
                        help="Posterior odds multiplier for any >=2-tag configuration versus a singlet in hierarchical/posterior/marginal decisions; 1 disables calibration")
    parser.add_argument("--signal-shape-min", type=float, default=1.0,
                        help="Lower bound on the shared NB signal shape r; increasing it lightens the low-count signal tail")
    parser.add_argument("--init-tail-fraction", type=float, default=None,
                        help="Optional fixed per-tag upper-tail fraction used only to seed NB signal parameters; default remains adaptive")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--configuration-prior", default="auto", choices=["auto", "enumerated", "factorized"],
                        help="Auto uses implicit all-order inference when full higher-order enumeration exceeds its budget")
    parser.add_argument("--omit-tag-marginals", action="store_true",
                        help="Omit per-tag posterior CSV columns; calls and confidence are unchanged")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def pick(name):
        plain = input_dir / name
        gz = input_dir / f"{name}.gz"
        if gz.exists():
            return gz
        if plain.exists():
            return plain
        raise FileNotFoundError(f"Missing {plain} or {gz}")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    script_start = time.time()
    method_start = time.time()

    manager = DemuxManager()
    counts, barcodes = manager.load_10x_mtx(
        pick("matrix.mtx"),
        pick("features.tsv"),
        pick("barcodes.tsv"),
    )
    prior_hint = args.configuration_prior if args.model_type in ['auto', 'NB', 'DLN'] and args.method == 'EM' else 'enumerated'
    if args.model_type == 'DLN':prior_hint = 'factorized'
    configs = manager.build_configs(max_klet=args.max_klet, max_auto_configs=args.max_auto_configs,
                                    configuration_prior=prior_hint)
    fit_options = {"batch_size": args.batch_size, "posterior_device": "cpu"} if args.method == "EM" else {}
    results = manager.run_demux(
        counts,
        model_type=args.model_type,
        method=args.method,
        device=device,
        max_iter=args.max_iter,
        configuration_prior=args.configuration_prior,
        signal_shape_min=args.signal_shape_min,
        init_tail_fraction=args.init_tail_fraction,
        **fit_options,
    )

    if 'factorized_model' in results:
        calls = results['factorized_model'].classify(
            counts, barcodes, results['tag_names'], batch_size=args.batch_size,
            posterior_thresh=args.posterior_thresh, map_thresh=args.map_thresh,
            decision_mode=args.decision_mode,
            marginal_singlet_thresh=args.marginal_singlet_thresh,
            marginal_doublet_thresh=args.marginal_doublet_thresh,
            multiplet_prior_scale=args.multiplet_prior_scale,
            include_tag_probabilities=not args.omit_tag_marginals)
    else:
        clf = HTClassifier(results["cfg_labels"], results["tag_names"], configs)
        calls = clf.classify(
            results["posteriors"],
            barcodes,
            map_thresh=args.map_thresh,
            decision_mode=args.decision_mode,
            marginal_singlet_thresh=args.marginal_singlet_thresh,
            marginal_doublet_thresh=args.marginal_doublet_thresh,
            marginal_batch_size=args.batch_size,
            compute_device=device,
            posterior_thresh=args.posterior_thresh,
            multiplet_prior_scale=args.multiplet_prior_scale,
        )

    if args.omit_tag_marginals:
        calls = calls.drop(columns=[c for c in calls if c.startswith('p_HTO')])

    method_runtime = time.time() - method_start
    calls.to_csv(output_dir / "HTDemux_classification.csv", index=False)

    summary = calls.rename(columns={"barcode": "droplet_id", "assignment_final": "final_assign"})
    summary.to_csv(output_dir / "HTDemux_summary.csv", index=False)

    runtime = pd.DataFrame([{
        "method": "HTDemux",
        "input_dir": str(input_dir),
        "n_hto": len(results["tag_names"]),
        "n_droplets": len(calls),
        "model_type": results['fit_diagnostics'].get('emission_family', 'NB' if args.model_type == 'auto' else args.model_type),
        "requested_model_type": args.model_type,
        "optimizer": args.method,
        "max_klet": manager.config_policy['resolved_max_klet'],
        "requested_max_klet": args.max_klet,
        "n_configs": manager.config_policy['n_configs'],
        "configuration_prior": results['fit_diagnostics'].get('configuration_prior', 'enumerated'),
        "n_configs_represented": manager.config_policy.get('n_configs_represented', str(len(configs))),
        "map_thresh": args.map_thresh,
        "decision_mode": args.decision_mode,
        "posterior_thresh": args.posterior_thresh,
        "multiplet_prior_scale": args.multiplet_prior_scale,
        "marginal_singlet_thresh": args.marginal_singlet_thresh,
        "marginal_doublet_thresh": args.marginal_doublet_thresh,
        "max_iter": args.max_iter,
        "batch_size": args.batch_size,
        "converged": results["fit_diagnostics"].get("converged"),
        "iterations": results["fit_diagnostics"].get("iterations"),
        "device": device,
        "method_runtime_sec": method_runtime,
        "total_runtime_sec": time.time() - script_start,
    }])
    runtime.to_csv(output_dir / "HTDemux_runtime.csv", index=False)
    (output_dir / "HTDemux_fit_diagnostics.json").write_text(json.dumps(results["fit_diagnostics"], indent=2))
    pd.DataFrame(results["fitted_parameters"], index=results["tag_names"]).to_csv(output_dir / "HTDemux_fitted_parameters.csv", index_label="tag")

    with open(output_dir / "HTDemux_config.json", "w") as fh:
        json.dump({**vars(args), "resolved_max_klet": manager.config_policy['resolved_max_klet'],
                   "n_configs": manager.config_policy['n_configs'],
                   "n_configs_represented": manager.config_policy.get('n_configs_represented', str(len(configs)))}, fh, indent=2)


if __name__ == "__main__":
    main()
