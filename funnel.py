import numpy as np
import pandas as pd
import warnings
import networkx as nx
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import normalized_mutual_info_score

def discretizar_numericas(
    df,
    numeric_cols,
    n_bins=4,
    strategy="quantile"
):
    df = df.copy()

    numeric_cols = numeric_cols or []

    if len(numeric_cols) == 0:
        return df

    imputer = SimpleImputer(strategy="median")
    X_num = imputer.fit_transform(df[numeric_cols])

    disc = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy=strategy
    )

    df[numeric_cols] = disc.fit_transform(X_num)

    for col in numeric_cols:
        df[col] = df[col].astype("category")

    return df



def preparar_X_para_mi_binaria(
    df,
    numeric_cols,
    categorical_cols
):
    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []

    partes = []
    discrete_features = []

    # Numéricas
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")

        X_num = pd.DataFrame(
            num_imputer.fit_transform(df[numeric_cols]),
            columns=numeric_cols,
            index=df.index
        )

        partes.append(X_num)
        discrete_features.extend([False] * len(numeric_cols))

    # Categóricas
    if len(categorical_cols) > 0:
        X_cat_raw = (
            df[categorical_cols]
            .astype("object")
            .fillna("__MISSING__")
        )

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

        X_cat = pd.DataFrame(
            encoder.fit_transform(X_cat_raw),
            columns=categorical_cols,
            index=df.index
        )

        partes.append(X_cat)
        discrete_features.extend([True] * len(categorical_cols))

    if len(partes) == 0:
        raise ValueError("No hay variables numéricas ni categóricas para evaluar.")

    X = pd.concat(partes, axis=1)

    return X, discrete_features



def relevance_via_binary_mi(
    df,
    target_col,
    numeric_cols=None,
    categorical_cols=None,
    random_state=123
):
    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []

    y = df[target_col].copy()

    # Eliminar filas con target nulo
    mask = y.notna()
    df = df.loc[mask].copy()
    y = y.loc[mask].copy()

    valores_target = sorted(pd.Series(y).unique())

    if len(valores_target) != 2:
        raise ValueError(
            f"El target debe ser binario. Valores encontrados: {valores_target}"
        )

    X, discrete_features = preparar_X_para_mi_binaria(
        df=df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols
    )

    mi_scores = mutual_info_classif(
        X,
        y,
        discrete_features=discrete_features,
        random_state=random_state
    )

    relevance = dict(zip(X.columns, mi_scores))

    return relevance



def nmi_mixed_matrix(
    df,
    numeric_cols=None,
    categorical_cols=None,
    n_bins=4,
    strategy="quantile"
):
    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []

    df_temp = df.copy()

    df_temp = discretizar_numericas(
        df=df_temp,
        numeric_cols=numeric_cols,
        n_bins=n_bins,
        strategy=strategy
    )

    for col in categorical_cols:
        df_temp[col] = (
            df_temp[col]
            .astype("object")
            .fillna("__MISSING__")
            .astype("category")
        )

    all_cols = numeric_cols + categorical_cols

    if len(all_cols) == 0:
        raise ValueError("No hay variables para calcular matriz NMI.")

    matriz = pd.DataFrame(
        np.zeros((len(all_cols), len(all_cols))),
        index=all_cols,
        columns=all_cols
    )

    for i in range(len(all_cols)):
        for j in range(i, len(all_cols)):
            nmi = normalized_mutual_info_score(
                df_temp[all_cols[i]],
                df_temp[all_cols[j]]
            )

            matriz.iloc[i, j] = nmi
            matriz.iloc[j, i] = nmi

    return matriz



def eliminar_redundancias_seguro(
    relevancia_dict,
    nmi_matrix,
    nmi_threshold=0.8,
    criterion="max"
):
    relevancia_limpia = {
        k: v for k, v in relevancia_dict.items()
        if not pd.isna(v)
    }

    vars_ = list(relevancia_limpia.keys())

    if len(vars_) == 0:
        return []

    G = nx.Graph()
    G.add_nodes_from(vars_)

    for i in range(len(vars_)):
        for j in range(i + 1, len(vars_)):
            var_i = vars_[i]
            var_j = vars_[j]

            if var_i in nmi_matrix.index and var_j in nmi_matrix.columns:
                if nmi_matrix.loc[var_i, var_j] >= nmi_threshold:
                    G.add_edge(var_i, var_j)

    grupos = list(nx.connected_components(G))

    if criterion == "max":
        selected = [
            max(grupo, key=lambda x: relevancia_limpia[x])
            for grupo in grupos
        ]

    elif criterion == "min":
        selected = [
            min(grupo, key=lambda x: relevancia_limpia[x])
            for grupo in grupos
        ]

    else:
        raise ValueError("criterion debe ser 'min' o 'max'")

    selected = sorted(
        selected,
        key=lambda x: relevancia_limpia[x],
        reverse=True
    )

    return selected



def binary_classification_funnel(
    df,
    target_col,
    numeric_cols=None,
    categorical_cols=None,
    n_bins=4,
    strategy="quantile",
    nmi_threshold=0.8,
    min_mi=0.0,
    top_k=None,
    random_state=123
):
    numeric_cols = numeric_cols or []
    categorical_cols = categorical_cols or []

    relevance = relevance_via_binary_mi(
        df=df,
        target_col=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        random_state=random_state
    )

    ranking = (
        pd.DataFrame({
            "variable": list(relevance.keys()),
            "mi_score": list(relevance.values())
        })
        .sort_values("mi_score", ascending=False)
        .reset_index(drop=True)
    )

    relevance_filtrada = {
        col: score
        for col, score in relevance.items()
        if score >= min_mi and not pd.isna(score)
    }

    if len(relevance_filtrada) == 0:
        warnings.warn("Ninguna variable superó el umbral mínimo de MI.")
        return [], ranking, None

    nmi_matrix = nmi_mixed_matrix(
        df=df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        n_bins=n_bins,
        strategy=strategy
    )

    selected = eliminar_redundancias_seguro(
        relevancia_dict=relevance_filtrada,
        nmi_matrix=nmi_matrix,
        nmi_threshold=nmi_threshold,
        criterion="max"
    )

    if top_k is not None:
        selected = selected[:top_k]

    return selected, ranking, nmi_matrix
