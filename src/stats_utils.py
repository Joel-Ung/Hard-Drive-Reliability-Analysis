"""
Statistical analysis utilities — Week 3.

Operates on the small one row per drive `cleaning.build_snapshot_dataset`. 
Since it is small and collapsed, pandas/scipy is sufficient here, and no DuckDB is needed at this stage.

Are failed drives significantly different from healthy drives, and can this difference be trusted?
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


def mann_whitney_by_attribute(df: pd.DataFrame, attributes: list, group_col: str = "label") -> pd.DataFrame:
    """
    Is each attribute different, one at a time?

    t-test assumes data is bell-curved shape, but SMART attributes may have long tail of high values.
    The Mann-Whitney U test do not compare based solely on raw average values. It ranks every value and compares between groups.
    This allows the test to be robust to outliers. 
    
    Run a Mann-Whitney U test per SMART attribute, comparing failed vs. healthy
    drives. Appropriate here since these attributes are rarely normally distributed
    and the two groups are very unequal in size.

    Also reports the rank-biserial correlation as an effect-size measure, since p-values 
    alone do not indicate whether a statistically significant difference is meaningful. 
    With an imbalanced dataset, tiny real differences can still produce very small p-values.

    The p-value indicates if there is a real difference, and the effect size indicates how big that difference is.
    """
    results = []
    n1 = (df[group_col] == 1).sum()
    n0 = (df[group_col] == 0).sum()

    for attr in attributes:
        group0 = df.loc[df[group_col] == 0, attr].dropna()
        group1 = df.loc[df[group_col] == 1, attr].dropna()
        if len(group0) == 0 or len(group1) == 0:
            continue
        stat, p = mannwhitneyu(group1, group0, alternative="two-sided")
        # rank-biserial effect size: 2U/(n1*n0) - 1, ranges -1 to 1
        effect = 2 * stat / (len(group1) * len(group0)) - 1
        results.append({
            "attribute": attr,
            "u_stat": stat,
            "p_value": p,
            "effect_size_rank_biserial": round(effect, 4),
            "healthy_median": group0.median(),
            "failed_median": group1.median(),
        })

    out = pd.DataFrame(results).sort_values("p_value").reset_index(drop=True)
    return out


def correlation_matrix(df: pd.DataFrame, attributes: list, method: str = "spearman") -> pd.DataFrame:
    """
    Pairwise correlation among SMART attributes, to flag multicollinearity before
    any modeling. Spearman (rank-based) by default, consistent with the
    non-normal, skewed nature of these attributes (same reasoning as the
    Mann-Whitney choice above).
    """
    return df[attributes].corr(method=method)


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Flag attribute pairs above a correlation threshold — candidates to consolidate."""
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({"attribute_1": cols[i], "attribute_2": cols[j], "correlation": round(val, 3)})
    if not pairs:
        return pd.DataFrame(columns=["attribute_1", "attribute_2", "correlation"])
    return pd.DataFrame(pairs).sort_values("correlation", key=abs, ascending=False).reset_index(drop=True)


def logistic_check(df: pd.DataFrame, features: list, target: str = "label",
                          test_size: float = 0.3, random_state: int = 42) -> dict:
    """
    A simple, non-tuned logistic regression to sanity-check whether there's
    separable signal in the top candidate features. This is descriptive only —
    not a final model, and not a substitute for Month 4's proper ML project.

    Uses class_weight="balanced" to compensate for the class imbalance rather
    than letting the model predict "healthy" for the majority of scenarios, and reports
    ROC-AUC rather than accuracy, which would be misleading on such an imbalanced dataset.
    """
    data = df[features + [target]].dropna()
    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    coef_table = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False).reset_index(drop=True)

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else None,
        "coefficients": coef_table,
        "classification_report": classification_report(y_test, y_pred, output_dict=False),
    }
