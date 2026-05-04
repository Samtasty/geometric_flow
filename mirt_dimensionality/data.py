"""
Data loading and batching for the M-IRT dimensionality experiment.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_pix(
    csv_path: str = "data/pix_data.csv",
    min_challenge_obs: int = 50,
    min_user_obs: int = 5,
) -> pd.DataFrame:
    """
    Load PIx data, keep only ok/ko outcomes, filter sparse challenges and users.
    Returns a DataFrame with columns: uid, cid, correct, competence_code, skill_id.
    """
    df = pd.read_csv(csv_path, index_col=0)
    df = df[df["answer_result"].isin(["ok", "ko"])].copy()
    df["correct"] = (df["answer_result"] == "ok").astype("int8")

    challenge_counts = df["challenge_id"].value_counts()
    valid_challenges = challenge_counts[challenge_counts >= min_challenge_obs].index
    df = df[df["challenge_id"].isin(valid_challenges)].copy()

    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_obs].index
    df = df[df["user_id"].isin(valid_users)].copy()

    user_enc = LabelEncoder().fit(df["user_id"])
    challenge_enc = LabelEncoder().fit(df["challenge_id"])
    df["uid"] = user_enc.transform(df["user_id"])
    df["cid"] = challenge_enc.transform(df["challenge_id"])

    print(
        f"Loaded {len(df):,} responses | "
        f"{df['uid'].nunique():,} users | "
        f"{df['cid'].nunique():,} challenges | "
        f"{df['competence_code'].nunique()} competences"
    )
    return df


def challenge_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame indexed by cid with competence_code and skill_id."""
    return (
        df.drop_duplicates("cid")[["cid", "challenge_id", "skill_id", "competence_code"]]
        .set_index("cid")
        .sort_index()
    )


def sample_student_batch(
    df: pd.DataFrame,
    n_students: int,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Sample n_students randomly and re-encode their uid to [0, n_students).
    Challenge cids are kept in their original global space.
    """
    rng = np.random.default_rng(seed)
    chosen = rng.choice(df["uid"].unique(), size=n_students, replace=False)
    df_batch = df[df["uid"].isin(set(chosen.tolist()))].copy()

    enc = LabelEncoder().fit(df_batch["uid"])
    df_batch["uid"] = enc.transform(df_batch["uid"])

    print(
        f"Batch: {len(df_batch):,} responses | "
        f"{df_batch['uid'].nunique():,} students | "
        f"{df_batch['cid'].nunique():,} challenges | "
        f"mean {len(df_batch)/df_batch['uid'].nunique():.1f} responses/student"
    )
    return df_batch


def response_split(df: pd.DataFrame, train_frac: float = 0.8, seed: int = 1):
    """Random response-level train/test split (not user-level)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(train_frac * len(idx))
    return df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()
