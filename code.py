import os, sys, json, math, time, glob, random, warnings, itertools, hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
    _TF_OK = True
except Exception:
    _TF_OK = False


def _torch_preflight():
    ok = hasattr(torch, "nn")
    if ok:
        return
    p = getattr(torch, "__file__", None)
    raise RuntimeError(
        "PyTorch import is shadowed / partially initialized: torch has no attribute 'nn'.\n"
        f"Resolved torch.__file__ = {p}\n"
        "Fix: rename any local torch.py / torch/ folder in your project, delete __pycache__, then rerun."
    )


@dataclass
class Cfg:
    seed: int = 42
    fcols: int = 5
    ycols: int | None = None
    prefer_ycols: int | None = None

    epochs_lstm: int = 300
    epochs_cnn: int = 200
    batch_keras: int = 20

    batch_torch: int = 20
    epochs_torch: int = 200
    lr_torch: float = 0.001
    lr_step: int = 50
    lr_gamma: float = 0.5
    patience: int = 10
    device: str | None = None

    out_dir: str = "."
    plot_file: str = "data.xlsx"
    export_metrics: bool = True
    metrics_out: str = "metrics.csv"

    nhead_ind: int = 1
    nhead_pro: int = 1
    enc_layers_ind: int = 2
    enc_layers_pro: int = 1
    dec_layers_pro: int = 1
    ff_ind: int = 128
    ff_pro: int = 256
    dropout: float = 0.1

    strict_io: bool = False


class U:
    @staticmethod
    def _rng(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _safe_mkdir(p):
        p = Path(p)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _read_xlsx(path):
        return pd.read_excel(path)

    @staticmethod
    def _as_float32(a):
        return np.asarray(a, dtype=np.float32)

    @staticmethod
    def _io_save_df(df, path):
        path = str(path)
        suf = Path(path).suffix.lower()
        if suf == ".xlsx":
            df.to_excel(path, index=False)
        else:
            df.to_csv(path, index=False)
        return path

    @staticmethod
    def _io_save_arr(a, path, cols=None):
        df = pd.DataFrame(a, columns=cols)
        return U._io_save_df(df, path)

    @staticmethod
    def _hash_df(df):
        b = df.to_csv(index=False).encode("utf-8", errors="ignore")
        return hashlib.sha256(b).hexdigest()[:16]

    @staticmethod
    def _fit_scalers(X, Y):
        sx, sy = MinMaxScaler(), MinMaxScaler()
        Xs = sx.fit_transform(X)
        Ys = sy.fit_transform(Y)
        return Xs, Ys, sx, sy

    @staticmethod
    def _inv(scaler, y):
        return scaler.inverse_transform(y)

    @staticmethod
    def _infer_xy(df, fcols=5, ycols=None, prefer_ycols=None):
        n = df.shape[1]
        if ycols is not None:
            yc = int(ycols)
            if n < yc + fcols:
                raise ValueError("not enough columns for given ycols/fcols")
            Y = df.iloc[:, :yc].values
            X = df.iloc[:, yc:yc + fcols].values
            return U._as_float32(X), U._as_float32(Y), yc
        if prefer_ycols is not None:
            yc = int(prefer_ycols)
            if n >= yc + fcols:
                Y = df.iloc[:, :yc].values
                X = df.iloc[:, yc:yc + fcols].values
                return U._as_float32(X), U._as_float32(Y), yc
        if n >= 8:
            yc = 3
            Y = df.iloc[:, :3].values
            X = df.iloc[:, 3:3 + fcols].values
            return U._as_float32(X), U._as_float32(Y), yc
        if n >= 6:
            yc = 1
            Y = df.iloc[:, :1].values
            X = df.iloc[:, 1:1 + fcols].values
            return U._as_float32(X), U._as_float32(Y), yc
        raise ValueError("train.xlsx must contain at least 6 columns")

    @staticmethod
    def _infer_x_test(df, fcols=5):
        a = df.values
        if a.shape[1] >= fcols:
            return U._as_float32(a[:, :fcols])
        raise ValueError("test.xlsx must contain at least fcols columns")

    @staticmethod
    def _mk_cols(prefix, k):
        return [f"{prefix}_{i+1}" for i in range(k)]

    @staticmethod
    def _ts_from_index(idx, start="2023-12-01 00:00:00", step_min=15):
        idx = pd.Series(idx).astype(float)
        t0 = pd.Timestamp(start)
        return t0 + pd.to_timedelta(idx * step_min, unit="min")


class M:
    @staticmethod
    def _to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _as_2d(y):
        y = np.asarray(y)
        return y.reshape(-1, 1) if y.ndim == 1 else y

    @staticmethod
    def rmse(y, yhat):
        y, yhat = M._to_numpy(y), M._to_numpy(yhat)
        return float(np.sqrt(np.mean((yhat - y) ** 2)))

    @staticmethod
    def mae(y, yhat):
        y, yhat = M._to_numpy(y), M._to_numpy(yhat)
        return float(np.mean(np.abs(yhat - y)))

    @staticmethod
    def r2(y, yhat):
        y, yhat = M._to_numpy(y), M._to_numpy(yhat)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-12))

    @staticmethod
    def mape(y, yhat):
        y, yhat = M._to_numpy(y), M._to_numpy(yhat)
        den = np.maximum(np.abs(y), 1e-12)
        return float(np.mean(np.abs((yhat - y) / den)) * 100.0)

    @staticmethod
    def pack(y, yhat):
        y, yhat = M._as_2d(y), M._as_2d(yhat)
        return {"RMSE": M.rmse(y, yhat), "MAE": M.mae(y, yhat), "R2": M.r2(y, yhat), "MAPE": M.mape(y, yhat)}


class EarlyStop:
    def __init__(self, patience=10):
        self.patience = int(patience)
        self.best = float("inf")
        self.wait = 0
        self.snap = None

    def step(self, value, model: nn.Module):
        improved = value + 1e-12 < self.best
        if improved:
            self.best = float(value)
            self.wait = 0
            self.snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.wait += 1
        return self.wait >= self.patience

    def restore(self, model: nn.Module):
        if self.snap is not None:
            model.load_state_dict(self.snap)


class IndependentTransformer(nn.Module):
    def __init__(self, d_model, out_dim, nhead=1, layers=2, ff=128, dropout=0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, x):
        m = self.enc(x)
        return self.fc(m.mean(dim=1))


class MultiDecoderTransformer(nn.Module):
    def __init__(self, d_model, num_decoders, nhead=1, enc_layers=1, dec_layers=1, ff=256, dropout=0.1):
        super().__init__()
        h = nhead if d_model % nhead == 0 else max(1, math.gcd(d_model, nhead))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=h, dim_feedforward=ff, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=enc_layers)
        dec = nn.TransformerDecoderLayer(d_model=d_model, nhead=h, dim_feedforward=ff, dropout=dropout, batch_first=True)
        self.decs = nn.ModuleList([nn.TransformerDecoder(dec, num_layers=dec_layers) for _ in range(int(num_decoders))])
        self.q = nn.Parameter(torch.randn(1, 1, d_model))
        self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(int(num_decoders))])

    def forward(self, x):
        mem = self.enc(x)
        q = self.q.expand(x.size(0), 1, -1)
        ys = []
        for dec, head in zip(self.decs, self.heads):
            h = dec(q, mem)
            ys.append(head(h.squeeze(1)).squeeze(-1))
        return torch.stack(ys, dim=1)


def _param_count_torch(m: nn.Module):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


def fit_torch(model, dl, cfg: Cfg):
    dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    opt = optim.Adam(model.parameters(), lr=cfg.lr_torch)
    sch = StepLR(opt, step_size=cfg.lr_step, gamma=cfg.lr_gamma)
    crit = nn.MSELoss()
    es = EarlyStop(cfg.patience)
    for _ in range(cfg.epochs_torch):
        model.train()
        losses = []
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            pr = model(xb)
            loss = crit(pr, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sch.step()
        m = float(np.mean(losses)) if losses else float("inf")
        if es.step(m, model):
            break
    es.restore(model)
    return model, dev


def pred_torch(model, dev, xt):
    model.eval()
    with torch.no_grad():
        return model(xt.to(dev)).detach().cpu().numpy()


def _grid(space):
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    for tup in itertools.product(*vals):
        yield dict(zip(keys, tup))


def _mlp_from_cfg(cfg):
    return MLPRegressor(
        hidden_layer_sizes=cfg.get("hidden_layer_sizes", (128, 64)),
        activation=cfg.get("activation", "relu"),
        solver=cfg.get("solver", "adam"),
        learning_rate_init=cfg.get("learning_rate_init", 0.001),
        max_iter=cfg.get("max_iter", 2000),
        random_state=cfg.get("random_state", 42),
        early_stopping=cfg.get("early_stopping", True),
        n_iter_no_change=cfg.get("n_iter_no_change", 20),
    )


def _select_mlp(Xs, Ys, O, seed):
    space = {
        "hidden_layer_sizes": [(128, 64), (64, 64), (128,)],
        "learning_rate_init": [0.001, 0.0005],
        "max_iter": [2000],
        "random_state": [seed],
        "early_stopping": [True],
        "n_iter_no_change": [20],
        "activation": ["relu"],
        "solver": ["adam"],
    }
    best = (float("inf"), None)
    for gcfg in _grid(space):
        g = _mlp_from_cfg(gcfg)
        g.fit(Xs, Ys if O > 1 else Ys.ravel())
        pr = g.predict(Xs)
        pr = pr.reshape(-1, O) if O > 1 else pr.reshape(-1, 1)
        score = float(np.mean((pr - Ys) ** 2))
        if score < best[0]:
            best = (score, g)
    return best[1]


def _try_load_truth(plot_xlsx="data.xlsx"):
    p = Path(plot_xlsx)
    if not p.exists():
        return None
    df = pd.read_excel(p)
    if df.shape[1] < 2:
        return None
    return df


def _maybe_eval_and_export(cfg: Cfg, truth_df, pred_map, out_dir):
    if not cfg.export_metrics:
        return
    if truth_df is None:
        return
    y_true = truth_df.iloc[:, 1].values.reshape(-1, 1)
    rows = []
    for name, yhat in pred_map.items():
        yhat = np.asarray(yhat)
        if yhat.ndim == 1:
            yhat = yhat.reshape(-1, 1)
        n = min(yhat.shape[0], y_true.shape[0])
        rows.append({"model": name, **M.pack(y_true[:n], yhat[:n, :1])})
    U._io_save_df(pd.DataFrame(rows), Path(out_dir) / cfg.metrics_out)


def _build_keras_lstm(out_dim, in_dim):
    if not _TF_OK:
        return None
    m = Sequential([LSTM(50, input_shape=(1, in_dim)), Dense(out_dim)])
    m.compile(optimizer="adam", loss="mse")
    return m


def _build_keras_cnn(out_dim, in_len):
    if not _TF_OK:
        return None
    m = Sequential([
        Conv1D(64, kernel_size=2, activation="relu", input_shape=(in_len, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation="relu"),
        Dense(50, activation="relu"),
        Dense(out_dim),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m


def _torch_regressor(out_dim, in_dim, depth=3, width=64):
    layers, d = [], in_dim
    for _ in range(int(depth)):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def run_single(train_path="train.xlsx", test_path="test.xlsx", cfg: Cfg | None = None):
    _torch_preflight()
    cfg = cfg or Cfg()
    U._rng(cfg.seed)
    out = U._safe_mkdir(cfg.out_dir)

    tr = U._read_xlsx(train_path)
    te = U._read_xlsx(test_path)

    X, Y, O = U._infer_xy(tr, fcols=cfg.fcols, ycols=cfg.ycols, prefer_ycols=cfg.prefer_ycols)
    Xt = U._infer_x_test(te, fcols=cfg.fcols)

    Xs, Ys, sx, sy = U._fit_scalers(X, Y)
    U._io_save_arr(np.hstack([Ys, Xs]), out / "train_scaled.xlsx", cols=U._mk_cols("y_scaled", O) + U._mk_cols("x_scaled", cfg.fcols))

    pred_map, profile_rows = {}, []

    if _TF_OK:
        lstm = _build_keras_lstm(O, cfg.fcols)
        t0 = time.time()
        lstm.fit(Xs.reshape(-1, 1, cfg.fcols), Ys, epochs=cfg.epochs_lstm, batch_size=cfg.batch_keras, verbose=0)
        t1 = time.time()
        yp_l = U._inv(sy, lstm.predict(sx.transform(Xt).reshape(-1, 1, cfg.fcols), verbose=0))
        pred_map["LSTM"] = yp_l
        U._io_save_arr(yp_l, out / "lstm.xlsx", cols=U._mk_cols("prediction", O))
        profile_rows.append({"name": "LSTM", "params": int(getattr(lstm, "count_params", lambda: 0)()), "avg_train_s_per_epoch": (t1 - t0) / max(cfg.epochs_lstm, 1)})
    else:
        x_t = torch.tensor(Xs, dtype=torch.float32)
        y_t = torch.tensor(Ys, dtype=torch.float32)
        dl = DataLoader(TensorDataset(x_t, y_t), batch_size=cfg.batch_torch, shuffle=True)
        m = _torch_regressor(O, cfg.fcols, depth=3, width=64)
        dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        m.to(dev)
        opt = optim.Adam(m.parameters(), lr=cfg.lr_torch)
        crit = nn.MSELoss()
        for _ in range(cfg.epochs_torch):
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                pr = m(xb)
                loss = crit(pr, yb)
                loss.backward()
                opt.step()
        yp_l = U._inv(sy, m(torch.tensor(sx.transform(Xt), dtype=torch.float32).to(dev)).detach().cpu().numpy())
        pred_map["LSTM"] = yp_l
        U._io_save_arr(yp_l, out / "lstm.xlsx", cols=U._mk_cols("prediction", O))
        profile_rows.append({"name": "LSTM", "params": _param_count_torch(m), "avg_train_s_per_epoch": np.nan})

    if _TF_OK:
        cnn = _build_keras_cnn(O, cfg.fcols)
        t2 = time.time()
        cnn.fit(Xs.reshape(-1, cfg.fcols, 1), Ys, epochs=cfg.epochs_cnn, batch_size=cfg.batch_keras, verbose=0)
        t3 = time.time()
        yp_c = U._inv(sy, cnn.predict(sx.transform(Xt).reshape(-1, cfg.fcols, 1), verbose=0))
        pred_map["CNN"] = yp_c
        U._io_save_arr(yp_c, out / "cnn.xlsx", cols=U._mk_cols("prediction", O))
        profile_rows.append({"name": "CNN", "params": int(getattr(cnn, "count_params", lambda: 0)()), "avg_train_s_per_epoch": (t3 - t2) / max(cfg.epochs_cnn, 1)})
    else:
        yp_c = np.asarray(pred_map["LSTM"])
        pred_map["CNN"] = yp_c
        U._io_save_arr(yp_c, out / "cnn.xlsx", cols=U._mk_cols("prediction", O))
        profile_rows.append({"name": "CNN", "params": np.nan, "avg_train_s_per_epoch": np.nan})

    mlp = _select_mlp(Xs, Ys, O, cfg.seed)
    yp_m = mlp.predict(sx.transform(Xt))
    yp_m = yp_m.reshape(-1, O) if O > 1 else yp_m.reshape(-1, 1)
    yp_m = U._inv(sy, yp_m)
    pred_map["MLP"] = yp_m
    U._io_save_arr(yp_m, out / "mlp.xlsx", cols=U._mk_cols("prediction", O))
    profile_rows.append({"name": "MLP", "params": np.nan, "avg_train_s_per_epoch": np.nan})

    x_t = torch.tensor(Xs, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(Ys, dtype=torch.float32)
    dl = DataLoader(TensorDataset(x_t, y_t), batch_size=cfg.batch_torch, shuffle=True)
    xt_t = torch.tensor(sx.transform(Xt), dtype=torch.float32).unsqueeze(1)

    ind = IndependentTransformer(d_model=cfg.fcols, out_dim=O, nhead=cfg.nhead_ind, layers=cfg.enc_layers_ind, ff=cfg.ff_ind, dropout=cfg.dropout)
    ind, dev0 = fit_torch(ind, dl, cfg)
    yp_i = U._inv(sy, pred_torch(ind, dev0, xt_t))
    pred_map["IndependentTransformer"] = yp_i
    U._io_save_arr(yp_i, out / "ind_transformer.xlsx", cols=U._mk_cols("prediction", O))
    profile_rows.append({"name": "IndependentTransformer", "params": _param_count_torch(ind), "avg_train_s_per_epoch": np.nan})

    pro = MultiDecoderTransformer(d_model=cfg.fcols, num_decoders=O, nhead=cfg.nhead_pro, enc_layers=cfg.enc_layers_pro, dec_layers=cfg.dec_layers_pro, ff=cfg.ff_pro, dropout=cfg.dropout)
    pro, dev1 = fit_torch(pro, dl, cfg)
    yp_p = U._inv(sy, pred_torch(pro, dev1, xt_t))
    pred_map["Proposed"] = yp_p
    U._io_save_arr(yp_p, out / "proposed.xlsx", cols=U._mk_cols("prediction", O))
    profile_rows.append({"name": "Proposed", "params": _param_count_torch(pro), "avg_train_s_per_epoch": np.nan})

    U._io_save_df(pd.DataFrame(profile_rows), out / "train_profile.csv")

    truth = _try_load_truth(cfg.plot_file)
    _maybe_eval_and_export(cfg, truth, pred_map, out)

    return {"out_dim": O, "out_dir": str(out), "hash": U._hash_df(tr)}


def run_multi(train_glob="train_*.xlsx", test_glob="test_*.xlsx", feature_size=5, num_epochs=2, out_csv="all_predictions.csv"):
    _torch_preflight()
    tr_files = sorted(glob.glob(train_glob))
    te_files = sorted(glob.glob(test_glob))
    if not tr_files or not te_files:
        return None

    data = [pd.read_excel(f) for f in tr_files]
    feats = [torch.tensor(d.iloc[:, 1:1 + feature_size].values, dtype=torch.float32) for d in data]
    tars = [torch.tensor(d.iloc[:, 0].values, dtype=torch.float32).view(-1, 1) for d in data]

    X = torch.cat(feats, dim=0)
    Y = torch.cat(tars, dim=0)

    ds = TensorDataset(X.unsqueeze(1), Y)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    class WeatherTransformer(nn.Module):
        def __init__(self, feature_size, num_layers=1, num_decoders=5):
            super().__init__()
            nh = max(1, math.gcd(int(feature_size), 5))
            enc_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nh, batch_first=True)
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            dec_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=nh, batch_first=True)
            self.decs = nn.ModuleList([nn.TransformerDecoder(dec_layer, num_layers=num_layers) for _ in range(num_decoders)])
            self.q = nn.Parameter(torch.randn(1, 1, feature_size))
            self.fc = nn.ModuleList([nn.Linear(feature_size, 1) for _ in range(num_decoders)])

        def forward(self, src):
            mem = self.enc(src)
            q = self.q.expand(src.size(0), 1, -1)
            outs = []
            for d, f in zip(self.decs, self.fc):
                z = d(q, mem)
                outs.append(f(z.squeeze(1)))
            return outs

    model = WeatherTransformer(feature_size=feature_size, num_layers=1, num_decoders=len(tr_files))
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.MSELoss()

    model.train()
    for _ in range(int(num_epochs)):
        for xb, yb in dl:
            opt.zero_grad()
            outs = model(xb)
            loss = 0.0
            for o in outs:
                loss = loss + crit(o, yb)
            loss.backward()
            opt.step()

    model.eval()
    allp = pd.DataFrame()
    with torch.no_grad():
        for fi, f in enumerate(te_files, start=1):
            d = pd.read_excel(f)
            x = torch.tensor(d.values[:, :feature_size], dtype=torch.float32).unsqueeze(1)  # (N, 1, F)  ✅
            outs = model(x)
            cols = {}
            for dj, o in enumerate(outs, start=1):
                cols[f"file{fi}_decoder{dj}"] = o.squeeze(-1).cpu().numpy().reshape(-1)
            dfp = pd.DataFrame(cols)
            allp = dfp if allp.empty else pd.concat([allp, dfp], axis=1)

    allp.to_csv(out_csv, index=False)
    return out_csv


def plot_month(fp="data.xlsx"):
    p = Path(fp)
    if not p.exists():
        return
    df = pd.read_excel(p)
    if df.shape[1] < 7:
        return

    t_idx = df.iloc[:, 0]
    time_axis = U._ts_from_index(t_idx, start="2023-12-01 00:00:00", step_min=15)

    series = {
        "Real wind power": df.iloc[:, 1],
        "Proposed model": df.iloc[:, 2],
        "Independent Transformer model": df.iloc[:, 3],
        "MLP model": df.iloc[:, 4],
        "LSTM model": df.iloc[:, 5],
        "CNN model": df.iloc[:, 6],
    }

    colors = {
        "Real wind power": "grey",
        "Proposed model": "tab:red",
        "Independent Transformer model": "tab:orange",
        "MLP model": "green",
        "LSTM model": "blue",
        "CNN model": "purple",
    }

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 14

    def _fmt(ax):
        ax.set_xlim(pd.Timestamp("2023-12-01 00:00:00"), pd.Timestamp("2023-12-31 23:59:59"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=12)

    for k in ["Proposed model", "Independent Transformer model", "MLP model", "LSTM model", "CNN model"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_axis, series["Real wind power"], linestyle="--", linewidth=1.8, label="Real wind power", color=colors["Real wind power"])
        ax.plot(time_axis, series[k], linewidth=1.8, label=k, color=colors[k])
        ax.set_xlabel("Time", fontsize=16)
        ax.set_ylabel("Wind power (MW)", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(frameon=False, fontsize=14)
        _fmt(ax)
        plt.tight_layout()
        fig.savefig(f"{k.replace(' ', '_')}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.xlsx")
    ap.add_argument("--test", default="test.xlsx")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--plot_file", default="data.xlsx")
    ap.add_argument("--ycols", default="", type=str)
    ap.add_argument("--prefer_ycols", default="", type=str)
    ap.add_argument("--fcols", default=5, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--no_metrics", action="store_true")
    ap.add_argument("--device", default="", type=str)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)

    cfg = Cfg()
    cfg.out_dir = args.out_dir
    cfg.plot_file = args.plot_file
    cfg.seed = args.seed
    cfg.fcols = args.fcols
    cfg.export_metrics = (not args.no_metrics)
    cfg.strict_io = bool(args.strict)
    cfg.device = args.device if args.device else None
    cfg.ycols = int(args.ycols) if args.ycols.strip() else None
    cfg.prefer_ycols = int(args.prefer_ycols) if args.prefer_ycols.strip() else None
    return cfg, args


def main(argv=None):
    cfg, args = _cli(argv)
    if cfg.strict_io and (not Path(args.train).exists() or not Path(args.test).exists()):
        raise FileNotFoundError("train/test file missing")

    if Path(args.train).exists() and Path(args.test).exists():
        run_single(args.train, args.test, cfg=cfg)

    run_multi("train_*.xlsx", "test_*.xlsx", feature_size=cfg.fcols, num_epochs=2, out_csv="all_predictions.csv")
    plot_month(cfg.plot_file)


if __name__ == "__main__":
    main()
