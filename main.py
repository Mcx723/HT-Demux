# main.py (with init_clusters integration)
from pathlib import Path
import torch
import pandas as pd
import numpy as np

from data import load_mtx, gen_configs
from model import NBModel, GMMModel
from train import run_em
from cluster import init_clusters   # <- 新增

# --- user-configurable ---
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)
INPUT_DIR = Path("10x")
KLET = 3
MODEL_TYPE = "GMM"                   # "NB" or "GMM"
MAP_THRESH = 0.9                     # MAP to Unassigned
MAX_ITER = 100
TOL = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------


def main():
    # 1) load HTO data
    counts, barcodes, tag_names = load_mtx(
        features_path=INPUT_DIR / "features.tsv.gz",
        barcodes_path=INPUT_DIR / "barcodes.tsv.gz",
        mtx_path=INPUT_DIR / "matrix.mtx.gz",
    )
    counts = counts.to(device=DEVICE)          # tensor float32
    N, H = counts.shape

    # 2) prepare X for models
    X_gmm = torch.log1p(counts)                # continuous input for GMM
    X_nb = counts.clone()                      # counts for NB

    # 3) configs + labels
    cfg_tensor, cfg_labels = gen_configs(H=H, klet=KLET, tag_names=tag_names)
    cfg_tensor = cfg_tensor.to(device=DEVICE, dtype=torch.float32)   # (K,H)
    K = cfg_tensor.size(0)

    # 4) choose model and data to use for clustering initialization
    if MODEL_TYPE.upper() == "GMM":
        model = GMMModel()
        X_model = X_gmm
        counts_model = None
        # sensible mu_sig for GMM: mean of X_gmm over cells
        mu_sig_for_init = X_gmm.mean(dim=0)
    else:
        model = NBModel()
        X_model = None
        counts_model = X_nb
        # sensible mu_sig for NB: mean counts per tag
        mu_sig_for_init = counts.mean(dim=0)

    # 5) use init_clusters to generate init_pi and init_post (gamma) when possible
    #    init_clusters requires a data tensor (X: (N,H)) -> choose X_model or counts_model accordingly
    init_pi = None
    init_post = None
    try:
        if X_model is not None:
            mu_init, pi_init, gamma_init = init_clusters(
                X=X_model,
                configs=cfg_tensor,
                mu_sig=mu_sig_for_init,
                n_subset=100,
                method="closest",
                return_gamma=True,
            )
        else:
            mu_init, pi_init, gamma_init = init_clusters(
                X=counts_model,
                configs=cfg_tensor,
                mu_sig=mu_sig_for_init,
                n_subset=100,
                method="closest",
                return_gamma=True,
            )

        # move to device and ensure dtypes
        init_pi = pi_init.to(device=DEVICE, dtype=torch.float32)
        if gamma_init is not None:
            init_post = gamma_init.to(device=DEVICE, dtype=torch.float32)
        else:
            init_post = None
    except Exception:
        # 保守回退：如果 init_clusters 出错（比如维度不匹配），不要阻塞主流程
        init_pi = None
        init_post = None

    # 6) init params (move to device)
    # Call model.init_params using keyword args so we don't pass None positionally.
    # This avoids Pylance complaining about Tensor | None -> Tensor mismatches.
    if X_model is not None:
        init_params = model.init_params(X=X_model)
    else:
        init_params = model.init_params(counts=counts_model)

    # ensure all returned params are on device
    init_params = {k: v.to(device=DEVICE) for k, v in init_params.items()}

    # 7) run EM (pass init_pi and init_post from init_clusters when available)
    em_out = run_em(
        model=model,
        X=X_model,
        counts=counts_model,
        configs=cfg_tensor,
        init_params=init_params,
        init_pi=init_pi,
        init_post=init_post,
        max_iter=MAX_ITER,
        tol=TOL
    )

    params = em_out["params"]
    pi = em_out["pi"].to(device=DEVICE)
    post = em_out["post"]
    # ensure post is a torch tensor on correct device
    if not torch.is_tensor(post):
        post = torch.tensor(post, dtype=torch.float32, device=DEVICE)
    else:
        post = post.to(device=DEVICE, dtype=torch.float32)

    # 8) compute evidence and per-tag marginals
    with torch.no_grad():
        loglik_nk = model.loglik_model(X_model, counts_model, params, configs=cfg_tensor)  # (N,K) on device
        logpi = torch.log(torch.clamp(pi, min=1e-12))
        lognum = loglik_nk + logpi.unsqueeze(0)     # (N,K)

        # stable evidence
        log_evidence = torch.logsumexp(lognum, dim=1)   # (N,)
        evidence = torch.exp(log_evidence).cpu().numpy()

        # posterior (recompute numerically stable)
        post_recomp = torch.exp(lognum - log_evidence.unsqueeze(1))
        post_recomp = post_recomp / (post_recomp.sum(dim=1, keepdim=True) + 1e-12)  # (N,K)

        # per-tag marginal p(tag=1) = post @ cfg  (post: N,K, cfg: K,H)
        per_tag_prob = post_recomp.matmul(cfg_tensor)   # (N,H)
        per_tag_prob = per_tag_prob.cpu().numpy()

        # type probabilities
        cfg_sums = cfg_tensor.sum(dim=1).cpu().numpy()  # (K,)
        none_mask = (cfg_sums == 0)
        singlet_mask = (cfg_sums == 1)
        multiplet_mask = (cfg_sums >= 2)

        none_prob = post_recomp[:, none_mask].sum(dim=1).cpu().numpy() if none_mask.any() else np.zeros(N)
        singlet_prob = post_recomp[:, singlet_mask].sum(dim=1).cpu().numpy() if singlet_mask.any() else np.zeros(N)
        multiplet_prob = post_recomp[:, multiplet_mask].sum(dim=1).cpu().numpy() if multiplet_mask.any() else np.zeros(N)

        # MAP assignments and probs
        map_idx = torch.argmax(post_recomp, dim=1).cpu().numpy()    # (N,)
        post_np = post_recomp.cpu().numpy()
        assigned_prob = post_np[np.arange(N), map_idx]

        # assigned label string: map cluster -> tag name(s) or "None"/"Multiplet"
        assigned_label = []
        # convert cfg_tensor to CPU numpy for simple indexing of which tags in cluster
        cfg_cpu = cfg_tensor.cpu().numpy()
        for k in map_idx:
            s = int(cfg_sums[k])
            if s == 0:
                assigned_label.append("None")
            elif s == 1:
                # find which tag index is 1 in cfg_cpu[k]
                ones = np.nonzero(cfg_cpu[k])[0]
                if ones.size == 1:
                    tag_idx = int(ones[0])
                    assigned_label.append(tag_names[tag_idx])
                else:
                    # unexpected, fallback to "Multiplet"
                    assigned_label.append("Multiplet")
            else:
                assigned_label.append("Multiplet")

        # apply MAP threshold to mark Unassigned
        assigned_label_thresh = [
            assigned_label[i] if assigned_prob[i] >= MAP_THRESH else "Unassigned"
            for i in range(N)
        ]

    # 9) assemble DataFrame (clean, standard columns)
    cols = {
        "droplet_id": barcodes,
        "assigned_cluster": [cfg_labels[int(k)] for k in map_idx],
        "assigned_label": assigned_label,
        "assigned_label_thresh": assigned_label_thresh,
        "assigned_prob": assigned_prob,
        "evidence": evidence,
        "none_prob": none_prob,
        "singlet_prob": singlet_prob,
        "multiplet_prob": multiplet_prob,
    }

    # add per-tag columns p_tag_<name>
    for j, name in enumerate(tag_names):
        cols[f"p_tag_{name}"] = per_tag_prob[:, j]

    df = pd.DataFrame(cols)

    # 10) save outputs
    out_csv = OUT_DIR / "HT-Demux_results.csv"
    df.to_csv(out_csv, index=False)

    # kept barcodes: singlets with assigned_prob >= threshold and not "None"
    kept_mask = [
        (assigned_label[i] != "Multiplet") and (assigned_label[i] != "None") and (assigned_label_thresh[i] != "Unassigned")
        for i in range(N)
    ]
    kept_barcodes = [b for b, keep in zip(barcodes, kept_mask) if keep]
    pd.Series(kept_barcodes).to_csv(OUT_DIR / "kept_barcodes.txt", index=False, header=False)

    print(f"Saved results to {out_csv}, kept {len(kept_barcodes)} barcodes")


if __name__ == "__main__":
    main()
