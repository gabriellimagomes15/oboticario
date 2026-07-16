print("ARQUIVO COM FUNÇÕES AUXILIARES")

# ============================================================
# BIBLIOTECAS
# ============================================================

import pandas as pd
from pathlib import Path
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import joblib


# ============================================================
# CONSTANTES / CONFIGURAÇÕES GERAIS
# ============================================================
ARQUIVO_BASE = "base-de-dados.xlsx"
NOME_SAIDA = "fase_1_auditoria_base.xlsx"

COL_SKU = "sk_produto_case"
COL_CICLO = "cod_ciclo"
COL_STATUS = "des_status_atual_agrup"
COL_PHASE_IN = "cod_phase_in_agrup"
COL_PHASE_OUT = "cod_phase_out_agrup"

COLS_META_POSSIVEL_LEAKAGE = [
    "des_status_atual_agrup",
    "cod_phase_out_agrup",
    "cod_ciclo_atual",
    "cod_ciclo_ultimo_considerado"
]

COLS_ESPERADAS = [
    "sk_produto_case",
    "cod_ciclo",
    "cod_ciclo_atual",
    "cod_ciclo_ultimo_considerado",
    "cod_canal_portfolio",
    "des_categoria_portfolio",
    "des_status_atual_agrup",
    "cod_phase_in_agrup",
    "cod_phase_out_agrup",
    "nr_dias_ciclo",
    "vlr_investimento_mkt_direto",
    "flg_repacking",
    "ind_qtd_ciclos_agrupador",
    "ind_cpfs_novos",
    "ind_cpfs_total",
    "tx_conversao_pdv",
    "ind_vlr_receita_real",
    "ind_vlr_receita_real_dia",
    "ind_vlr_receita_real_corrigido",
    "ind_vlr_receita_real_dia_corrigido",
    "ind_vlr_ruptura",
    "ind_vlr_baseline_dia_corrigido"
]

# ============================================================
# 1. FUNÇÕES AUXILIARES
# ============================================================

def carregar_base(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega a base Excel e padroniza nomes de colunas.
    """
    df = pd.read_excel(caminho_arquivo,
                       #dtype= {'ind_qtd_ciclos_agrupador':str},
                       engine="openpyxl")

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.replace(" ", "_")
        .str.lower()
    )

    return df


def criar_rank_ciclos(df: pd.DataFrame) -> dict:
    """
    Cria um ranking temporal dos ciclos disponíveis.

    Como cod_ciclo parece representar ano + ciclo comercial,
    e não necessariamente data mensal convencional, usamos a ordem
    dos ciclos existentes na própria base.
    """
    colunas_ciclo = [
        c for c in [
            COL_CICLO,
            "cod_ciclo_atual",
            "cod_ciclo_ultimo_considerado",
            COL_PHASE_IN,
            COL_PHASE_OUT
        ]
        if c in df.columns
    ]

    valores = []

    for col in colunas_ciclo:
        serie = pd.to_numeric(df[col], errors="coerce")
        serie = serie.dropna()
        serie = serie[serie != 999999]
        valores.extend(serie.unique().tolist())

    ciclos_ordenados = sorted(pd.Series(valores).dropna().unique())

    ciclo_rank = {
        ciclo: idx
        for idx, ciclo in enumerate(ciclos_ordenados)
    }

    return ciclo_rank


def adicionar_rank_temporal(df: pd.DataFrame, ciclo_rank: dict) -> pd.DataFrame:
    """
    Adiciona colunas com ranking temporal dos ciclos.
    """
    df = df.copy()

    if COL_CICLO in df.columns:
        df["rank_ciclo"] = pd.to_numeric(df[COL_CICLO], errors="coerce").map(ciclo_rank)

    if COL_PHASE_IN in df.columns:
        df["rank_phase_in"] = pd.to_numeric(df[COL_PHASE_IN], errors="coerce").map(ciclo_rank)

    if COL_PHASE_OUT in df.columns:
        df["rank_phase_out"] = pd.to_numeric(df[COL_PHASE_OUT], errors="coerce").map(ciclo_rank)

    return df


def classificar_tipo_variavel(serie: pd.Series) -> str:
    """
    Classifica o tipo da variável para facilitar leitura do relatório.
    """
    if pd.api.types.is_numeric_dtype(serie):
        return "numérica"
    elif pd.api.types.is_datetime64_any_dtype(serie):
        return "data"
    else:
        return "categórica/texto"


# ============================================================
# 2. ETAPA 1.1 — ESTRUTURA DOS DADOS
# ============================================================

def auditar_estrutura(df: pd.DataFrame) -> dict:
    """
    Audita estrutura geral da base:
    - dimensões
    - colunas
    - tipos
    - nulos
    - cardinalidade
    - chave SKU + ciclo
    - histórico temporal
    """
    resultados = {}

    resumo_geral = pd.DataFrame({
        "indicador": [
            "quantidade_linhas",
            "quantidade_colunas",
            "quantidade_skus_unicos",
            "quantidade_ciclos_unicos",
            "primeiro_ciclo",
            "ultimo_ciclo"
        ],
        "valor": [
            len(df),
            df.shape[1],
            df[COL_SKU].nunique() if COL_SKU in df.columns else np.nan,
            df[COL_CICLO].nunique() if COL_CICLO in df.columns else np.nan,
            df[COL_CICLO].min() if COL_CICLO in df.columns else np.nan,
            df[COL_CICLO].max() if COL_CICLO in df.columns else np.nan
        ]
    })

    perfil_colunas = []

    for col in df.columns:
        serie = df[col]

        perfil_colunas.append({
            "coluna": col,
            "tipo_pandas": str(serie.dtype),
            "tipo_interpretado": classificar_tipo_variavel(serie),
            "qtd_nulos": serie.isna().sum(),
            "perc_nulos": serie.isna().mean(),
            "qtd_unicos": serie.nunique(dropna=True),
            "perc_unicos": serie.nunique(dropna=True) / len(df),
            "exemplo_1": serie.dropna().iloc[0] if serie.dropna().shape[0] > 0 else np.nan,
            "exemplo_2": serie.dropna().iloc[1] if serie.dropna().shape[0] > 1 else np.nan,
            "exemplo_3": serie.dropna().iloc[2] if serie.dropna().shape[0] > 2 else np.nan
        })

    perfil_colunas = pd.DataFrame(perfil_colunas)

    colunas_ausentes = [
        col for col in COLS_ESPERADAS
        if col not in df.columns
    ]

    colunas_extras = [
        col for col in df.columns
        if col not in COLS_ESPERADAS
    ]

    validacao_colunas = pd.DataFrame({
        "tipo": ["colunas_ausentes", "colunas_extras"],
        "colunas": [", ".join(colunas_ausentes), ", ".join(colunas_extras)],
        "qtd": [len(colunas_ausentes), len(colunas_extras)]
    })

    if COL_SKU in df.columns and COL_CICLO in df.columns:
        duplicadas = (
            df
            .groupby([COL_SKU, COL_CICLO])
            .size()
            .reset_index(name="qtd_registros")
            .query("qtd_registros > 1")
        )

        resumo_chave = pd.DataFrame({
            "indicador": [
                "chave_testada",
                "qtd_combinacoes_duplicadas",
                "qtd_linhas_em_duplicidade"
            ],
            "valor": [
                f"{COL_SKU} + {COL_CICLO}",
                len(duplicadas),
                duplicadas["qtd_registros"].sum() if len(duplicadas) > 0 else 0
            ]
        })

        historico_sku = (
            df
            .groupby(COL_SKU)
            .agg(
                primeiro_ciclo=(COL_CICLO, "min"),
                ultimo_ciclo=(COL_CICLO, "max"),
                qtd_ciclos=(COL_CICLO, "nunique"),
                qtd_registros=(COL_CICLO, "size")
            )
            .reset_index()
        )

        resumo_historico = historico_sku["qtd_ciclos"].describe().reset_index()
        resumo_historico.columns = ["metrica", "valor"]

    else:
        duplicadas = pd.DataFrame()
        resumo_chave = pd.DataFrame()
        historico_sku = pd.DataFrame()
        resumo_historico = pd.DataFrame()

    resultados["resumo_geral"] = resumo_geral
    resultados["perfil_colunas"] = perfil_colunas
    resultados["validacao_colunas"] = validacao_colunas
    resultados["duplicidades_chave"] = duplicadas
    resultados["resumo_chave"] = resumo_chave
    resultados["historico_sku"] = historico_sku
    resultados["resumo_historico_sku"] = resumo_historico

    return resultados


# ============================================================
# 3. ETAPA 1.2 — AUDITORIA DO CICLO DE VIDA
# ============================================================

def auditar_ciclo_vida(df: pd.DataFrame) -> dict:
    """
    Audita status, phase-in e phase-out.
    Verifica se status varia no tempo e se phase-out parece ser informação futura.
    """
    resultados = {}

    # -----------------------------
    # Status por SKU
    # -----------------------------
    if COL_SKU in df.columns and COL_STATUS in df.columns:
        status_por_sku = (
            df
            .groupby(COL_SKU)
            .agg(
                qtd_status_distintos=(COL_STATUS, "nunique"),
                status_observados=(COL_STATUS, lambda x: " | ".join(sorted(x.dropna().astype(str).unique()))),
                primeiro_status=(COL_STATUS, lambda x: x.dropna().iloc[0] if x.dropna().shape[0] > 0 else np.nan),
                ultimo_status=(COL_STATUS, lambda x: x.dropna().iloc[-1] if x.dropna().shape[0] > 0 else np.nan),
                primeiro_ciclo=(COL_CICLO, "min"),
                ultimo_ciclo=(COL_CICLO, "max")
            )
            .reset_index()
        )

        status_por_sku["status_muda_no_tempo"] = status_por_sku["qtd_status_distintos"] > 1

        resumo_status_sku = pd.DataFrame({
            "indicador": [
                "qtd_skus",
                "skus_com_status_unico",
                "skus_com_status_variando_no_tempo",
                "perc_skus_com_status_variando_no_tempo"
            ],
            "valor": [
                len(status_por_sku),
                (status_por_sku["qtd_status_distintos"] == 1).sum(),
                status_por_sku["status_muda_no_tempo"].sum(),
                status_por_sku["status_muda_no_tempo"].mean()
            ]
        })

        dist_status_linhas = (
            df[COL_STATUS]
            .value_counts(dropna=False)
            .reset_index()
        )
        dist_status_linhas.columns = ["status", "qtd_linhas"]
        dist_status_linhas["perc_linhas"] = dist_status_linhas["qtd_linhas"] / len(df)

    else:
        status_por_sku = pd.DataFrame()
        resumo_status_sku = pd.DataFrame()
        dist_status_linhas = pd.DataFrame()

    # -----------------------------
    # Phase-out
    # -----------------------------
    if COL_PHASE_OUT in df.columns:
        df_phase = df.copy()
        df_phase[COL_PHASE_OUT] = pd.to_numeric(df_phase[COL_PHASE_OUT], errors="coerce")

        df_phase["phase_out_e_999999"] = df_phase[COL_PHASE_OUT] == 999999
        df_phase["tem_phase_out_real"] = (
            df_phase[COL_PHASE_OUT].notna()
            & (df_phase[COL_PHASE_OUT] != 999999)
        )

        dist_phase_out = (
            df_phase[COL_PHASE_OUT]
            .value_counts(dropna=False)
            .reset_index()
        )
        dist_phase_out.columns = ["cod_phase_out_agrup", "qtd_linhas"]
        dist_phase_out["perc_linhas"] = dist_phase_out["qtd_linhas"] / len(df_phase)

        if COL_STATUS in df_phase.columns:
            consistencia_status_phaseout = (
                df_phase
                .groupby(COL_STATUS)
                .agg(
                    qtd_linhas=(COL_STATUS, "size"),
                    qtd_phase_out_999999=("phase_out_e_999999", "sum"),
                    qtd_phase_out_real=("tem_phase_out_real", "sum")
                )
                .reset_index()
            )

            consistencia_status_phaseout["perc_phase_out_999999"] = (
                consistencia_status_phaseout["qtd_phase_out_999999"]
                / consistencia_status_phaseout["qtd_linhas"]
            )

            consistencia_status_phaseout["perc_phase_out_real"] = (
                consistencia_status_phaseout["qtd_phase_out_real"]
                / consistencia_status_phaseout["qtd_linhas"]
            )

        else:
            consistencia_status_phaseout = pd.DataFrame()

        if COL_SKU in df_phase.columns:
            phaseout_por_sku = (
                df_phase
                .groupby(COL_SKU)
                .agg(
                    qtd_phase_out_distintos=(COL_PHASE_OUT, "nunique"),
                    phase_out_observados=(COL_PHASE_OUT, lambda x: " | ".join(sorted(x.dropna().astype(str).unique()))),
                    menor_phase_out=(COL_PHASE_OUT, "min"),
                    maior_phase_out=(COL_PHASE_OUT, "max"),
                    tem_phase_out_real=("tem_phase_out_real", "max"),
                    qtd_linhas=(COL_PHASE_OUT, "size")
                )
                .reset_index()
            )

            phaseout_por_sku["phase_out_muda_no_tempo"] = (
                phaseout_por_sku["qtd_phase_out_distintos"] > 1
            )
        else:
            phaseout_por_sku = pd.DataFrame()

    else:
        dist_phase_out = pd.DataFrame()
        consistencia_status_phaseout = pd.DataFrame()
        phaseout_por_sku = pd.DataFrame()

    # -----------------------------
    # Consistência temporal phase-in / phase-out
    # -----------------------------
    if "rank_phase_in" in df.columns and "rank_phase_out" in df.columns:
        inconsistencias_phase = df.copy()

        inconsistencias_phase["phase_out_antes_do_phase_in"] = (
            inconsistencias_phase["rank_phase_out"].notna()
            & inconsistencias_phase["rank_phase_in"].notna()
            & (inconsistencias_phase["rank_phase_out"] < inconsistencias_phase["rank_phase_in"])
        )

        inconsistencias_phase_resumo = pd.DataFrame({
            "indicador": [
                "linhas_com_phase_out_antes_do_phase_in",
                "perc_linhas_com_phase_out_antes_do_phase_in"
            ],
            "valor": [
                inconsistencias_phase["phase_out_antes_do_phase_in"].sum(),
                inconsistencias_phase["phase_out_antes_do_phase_in"].mean()
            ]
        })

        linhas_phase_inconsistente = (
            inconsistencias_phase
            .loc[
                inconsistencias_phase["phase_out_antes_do_phase_in"],
                [c for c in [
                    COL_SKU,
                    COL_CICLO,
                    COL_PHASE_IN,
                    COL_PHASE_OUT,
                    "rank_phase_in",
                    "rank_phase_out",
                    COL_STATUS
                ] if c in inconsistencias_phase.columns]
            ]
            .head(1000)
        )

    else:
        inconsistencias_phase_resumo = pd.DataFrame()
        linhas_phase_inconsistente = pd.DataFrame()

    resultados["status_por_sku"] = status_por_sku
    resultados["resumo_status_sku"] = resumo_status_sku
    resultados["dist_status_linhas"] = dist_status_linhas
    resultados["dist_phase_out"] = dist_phase_out
    resultados["consistencia_status_phaseout"] = consistencia_status_phaseout
    resultados["phaseout_por_sku"] = phaseout_por_sku
    resultados["inconsistencias_phase_resumo"] = inconsistencias_phase_resumo
    resultados["linhas_phase_inconsistente"] = linhas_phase_inconsistente

    return resultados


# ============================================================
# 4. ETAPA 1.3 — QUALIDADE DOS DADOS
# ============================================================

def auditar_qualidade(df: pd.DataFrame) -> dict:
    """
    Audita qualidade geral:
    - nulos
    - negativos
    - zeros
    - outliers por regra IQR
    - variáveis constantes
    - variáveis suspeitas
    """
    resultados = {}

    # -----------------------------
    # Nulos
    # -----------------------------
    nulos = []

    for col in df.columns:
        nulos.append({
            "coluna": col,
            "qtd_nulos": df[col].isna().sum(),
            "perc_nulos": df[col].isna().mean()
        })

    nulos = pd.DataFrame(nulos).sort_values("perc_nulos", ascending=False)

    # -----------------------------
    # Variáveis constantes
    # -----------------------------
    constantes = []

    for col in df.columns:
        constantes.append({
            "coluna": col,
            "qtd_unicos": df[col].nunique(dropna=True),
            "variavel_constante": df[col].nunique(dropna=True) <= 1
        })

    constantes = pd.DataFrame(constantes)

    # -----------------------------
    # Numéricas: negativos, zeros e outliers
    # -----------------------------
    cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    qualidade_numerica = []
    outliers_detalhe = []

    for col in cols_numericas:
        serie = pd.to_numeric(df[col], errors="coerce")

        qtd_validos = serie.notna().sum()
        qtd_negativos = (serie < 0).sum()
        qtd_zeros = (serie == 0).sum()

        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)
        iqr = q3 - q1

        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr

        outlier_mask = (serie < limite_inf) | (serie > limite_sup)
        qtd_outliers = outlier_mask.sum()

        qualidade_numerica.append({
            "coluna": col,
            "qtd_validos": qtd_validos,
            "min": serie.min(),
            "p01": serie.quantile(0.01),
            "p05": serie.quantile(0.05),
            "media": serie.mean(),
            "mediana": serie.median(),
            "p95": serie.quantile(0.95),
            "p99": serie.quantile(0.99),
            "max": serie.max(),
            "qtd_negativos": qtd_negativos,
            "perc_negativos": qtd_negativos / len(df),
            "qtd_zeros": qtd_zeros,
            "perc_zeros": qtd_zeros / len(df),
            "limite_outlier_inferior_iqr": limite_inf,
            "limite_outlier_superior_iqr": limite_sup,
            "qtd_outliers_iqr": qtd_outliers,
            "perc_outliers_iqr": qtd_outliers / len(df)
        })

        if qtd_outliers > 0:
            temp = df.loc[outlier_mask, [c for c in [COL_SKU, COL_CICLO, col] if c in df.columns]].copy()
            temp["coluna_outlier"] = col
            temp["valor_outlier"] = serie[outlier_mask]
            outliers_detalhe.append(temp.head(200))

    qualidade_numerica = pd.DataFrame(qualidade_numerica)

    if len(outliers_detalhe) > 0:
        outliers_detalhe = pd.concat(outliers_detalhe, ignore_index=True)
    else:
        outliers_detalhe = pd.DataFrame()

    # -----------------------------
    # flg_repacking
    # -----------------------------
    if "flg_repacking" in df.columns:
        valores_repacking = (
            df["flg_repacking"]
            .value_counts(dropna=False)
            .reset_index()
        )
        valores_repacking.columns = ["valor_flg_repacking", "qtd_linhas"]
        valores_repacking["perc_linhas"] = valores_repacking["qtd_linhas"] / len(df)

        valores_unicos = set(pd.to_numeric(df["flg_repacking"], errors="coerce").dropna().unique())
        valores_binarios_esperados = {0, 1}

        repacking_resumo = pd.DataFrame({
            "indicador": [
                "qtd_valores_unicos",
                "parece_binaria_0_1",
                "menor_valor",
                "maior_valor"
            ],
            "valor": [
                df["flg_repacking"].nunique(dropna=True),
                valores_unicos.issubset(valores_binarios_esperados),
                pd.to_numeric(df["flg_repacking"], errors="coerce").min(),
                pd.to_numeric(df["flg_repacking"], errors="coerce").max()
            ]
        })
    else:
        valores_repacking = pd.DataFrame()
        repacking_resumo = pd.DataFrame()

    resultados["nulos"] = nulos
    resultados["constantes"] = constantes
    resultados["qualidade_numerica"] = qualidade_numerica
    resultados["outliers_detalhe"] = outliers_detalhe
    resultados["valores_repacking"] = valores_repacking
    resultados["repacking_resumo"] = repacking_resumo

    return resultados


# ============================================================
# 5. ETAPA 1.4 — AUDITORIA DE VAZAMENTO TEMPORAL
# ============================================================

def auditar_leakage(df: pd.DataFrame, ciclo_vida: dict) -> dict:
    """
    Classifica variáveis com possível risco de vazamento temporal.
    """
    resultados = {}

    linhas = []

    for col in df.columns:
        if col == COL_PHASE_OUT:
            risco = "alto"
            motivo = (
                "Pode representar informação futura de encerramento do SKU. "
                "Não deve ser usada como feature se o objetivo for prever phase-out."
            )
            uso_recomendado = "usar apenas para construir target ou análise de ciclo de vida"

        elif col == COL_STATUS:
            risco = "alto/médio"
            motivo = (
                "Pode representar status atual replicado no histórico. "
                "Se não variar temporalmente, entrega o futuro ao modelo."
            )
            uso_recomendado = "validar antes; provavelmente não usar como feature"

        elif col in ["cod_ciclo_atual", "cod_ciclo_ultimo_considerado"]:
            risco = "médio/alto"
            motivo = (
                "Pode representar metadado do recorte da base, não uma informação disponível "
                "no ciclo histórico observado."
            )
            uso_recomendado = "não usar como feature até confirmar semântica"

        elif col == COL_PHASE_IN:
            risco = "baixo/médio"
            motivo = (
                "Phase-in pode ser conhecido desde o início do produto. "
                "Pode ser útil para calcular idade, mas precisa ser tratado com cuidado."
            )
            uso_recomendado = "usar preferencialmente para derivar idade do SKU"

        elif col in ["rank_phase_out"]:
            risco = "alto"
            motivo = "Derivado diretamente de phase-out."
            uso_recomendado = "não usar como feature"

        elif col in ["rank_phase_in"]:
            risco = "baixo/médio"
            motivo = "Derivado de phase-in."
            uso_recomendado = "usar com cautela para idade"

        elif col in ["rank_ciclo"]:
            risco = "baixo"
            motivo = "Representa a posição temporal do ciclo observado."
            uso_recomendado = "pode ser usado para ordenação temporal, não necessariamente como preditor direto"

        else:
            risco = "baixo/avaliar"
            motivo = "Não parece entregar diretamente informação futura, mas deve ser avaliada no contexto temporal."
            uso_recomendado = "pode ser candidata a feature após validação"

        linhas.append({
            "coluna": col,
            "risco_leakage": risco,
            "motivo": motivo,
            "uso_recomendado": uso_recomendado
        })

    leakage_colunas = pd.DataFrame(linhas)

    # Diagnóstico específico do status
    if "status_por_sku" in ciclo_vida and not ciclo_vida["status_por_sku"].empty:
        status_por_sku = ciclo_vida["status_por_sku"]

        diagnostico_status = pd.DataFrame({
            "indicador": [
                "qtd_skus_status_nao_muda",
                "qtd_skus_status_muda",
                "perc_skus_status_muda"
            ],
            "valor": [
                (status_por_sku["status_muda_no_tempo"] == False).sum(),
                (status_por_sku["status_muda_no_tempo"] == True).sum(),
                status_por_sku["status_muda_no_tempo"].mean()
            ]
        })
    else:
        diagnostico_status = pd.DataFrame()

    resultados["leakage_colunas"] = leakage_colunas
    resultados["diagnostico_status"] = diagnostico_status

    return resultados


# ============================================================
# 6. RECOMENDAÇÕES AUTOMÁTICAS
# ============================================================

def gerar_recomendacoes(estrutura: dict, ciclo_vida: dict, qualidade: dict, leakage: dict) -> pd.DataFrame:
    """
    Gera recomendações iniciais baseadas nas auditorias.
    """
    recomendacoes = []

    # Chave
    if "duplicidades_chave" in estrutura:
        qtd_dup = len(estrutura["duplicidades_chave"])

        if qtd_dup > 0:
            recomendacoes.append({
                "tema": "Chave de negócio",
                "severidade": "alta",
                "recomendacao": (
                    "Existem combinações duplicadas de SKU + ciclo. "
                    "Investigar antes de modelar, pois uma linha deveria representar um SKU em um ciclo."
                )
            })
        else:
            recomendacoes.append({
                "tema": "Chave de negócio",
                "severidade": "baixa",
                "recomendacao": "A combinação SKU + ciclo não apresentou duplicidades."
            })

    # Status
    if "resumo_status_sku" in ciclo_vida and not ciclo_vida["resumo_status_sku"].empty:
        resumo_status = ciclo_vida["resumo_status_sku"]

        perc_muda = resumo_status.loc[
            resumo_status["indicador"] == "perc_skus_com_status_variando_no_tempo",
            "valor"
        ]

        if len(perc_muda) > 0 and perc_muda.iloc[0] == 0:
            recomendacoes.append({
                "tema": "Status atual",
                "severidade": "alta",
                "recomendacao": (
                    "O status não parece variar ao longo do tempo. "
                    "Isso indica forte risco de vazamento temporal caso seja usado como feature."
                )
            })
        else:
            recomendacoes.append({
                "tema": "Status atual",
                "severidade": "média",
                "recomendacao": (
                    "O status apresenta alguma variação temporal ou precisa ser analisado em detalhe. "
                    "Validar se representa status histórico ou status atual replicado."
                )
            })

    # Phase-out
    recomendacoes.append({
        "tema": "Phase-out",
        "severidade": "alta",
        "recomendacao": (
            "A coluna de phase-out deve ser tratada como candidata para construção do target, "
            "mas não deve ser usada como variável explicativa do modelo."
        )
    })

    # Repacking
    if "repacking_resumo" in qualidade and not qualidade["repacking_resumo"].empty:
        repacking = qualidade["repacking_resumo"]

        parece_binaria = repacking.loc[
            repacking["indicador"] == "parece_binaria_0_1",
            "valor"
        ]

        if len(parece_binaria) > 0 and parece_binaria.iloc[0] is False:
            recomendacoes.append({
                "tema": "flg_repacking",
                "severidade": "alta",
                "recomendacao": (
                    "A variável flg_repacking não parece binária, apesar do nome sugerir uma flag. "
                    "Validar documentação, ordem das colunas ou significado real do campo."
                )
            })

    # Target futuro
    recomendacoes.append({
        "tema": "Definição de target",
        "severidade": "alta",
        "recomendacao": (
            "Após esta auditoria, a próxima fase deve definir se o alvo será: "
            "'entrará em phase-out nos próximos 25 ciclos' ou outra definição operacional de declínio."
        )
    })

    # Leakage
    recomendacoes.append({
        "tema": "Leakage temporal",
        "severidade": "alta",
        "recomendacao": (
            "Antes da modelagem, separar claramente variáveis disponíveis no ciclo observado "
            "de variáveis conhecidas apenas após a descontinuação."
        )
    })

    return pd.DataFrame(recomendacoes)


# ============================================================
# 7. EXPORTAÇÃO DO RELATÓRIO
# ============================================================

def exportar_relatorio_excel(
    nome_saida: str,
    estrutura: dict,
    ciclo_vida: dict,
    qualidade: dict,
    leakage: dict,
    recomendacoes: pd.DataFrame
):
    """
    Exporta todas as tabelas para um arquivo Excel com múltiplas abas.
    """
    with pd.ExcelWriter(nome_saida, engine="openpyxl") as writer:

        # Estrutura
        estrutura["resumo_geral"].to_excel(writer, sheet_name="01_resumo_geral", index=False)
        estrutura["perfil_colunas"].to_excel(writer, sheet_name="02_perfil_colunas", index=False)
        estrutura["validacao_colunas"].to_excel(writer, sheet_name="03_validacao_colunas", index=False)
        estrutura["resumo_chave"].to_excel(writer, sheet_name="04_resumo_chave", index=False)
        estrutura["duplicidades_chave"].to_excel(writer, sheet_name="05_duplicidades_chave", index=False)
        estrutura["resumo_historico_sku"].to_excel(writer, sheet_name="06_resumo_hist_sku", index=False)
        estrutura["historico_sku"].to_excel(writer, sheet_name="07_historico_sku", index=False)

        # Ciclo de vida
        ciclo_vida["resumo_status_sku"].to_excel(writer, sheet_name="08_resumo_status", index=False)
        ciclo_vida["dist_status_linhas"].to_excel(writer, sheet_name="09_dist_status", index=False)
        ciclo_vida["status_por_sku"].to_excel(writer, sheet_name="10_status_por_sku", index=False)
        ciclo_vida["dist_phase_out"].to_excel(writer, sheet_name="11_dist_phaseout", index=False)
        ciclo_vida["consistencia_status_phaseout"].to_excel(writer, sheet_name="12_status_phaseout", index=False)
        ciclo_vida["phaseout_por_sku"].to_excel(writer, sheet_name="13_phaseout_por_sku", index=False)
        ciclo_vida["inconsistencias_phase_resumo"].to_excel(writer, sheet_name="14_resumo_phase_in_out", index=False)
        ciclo_vida["linhas_phase_inconsistente"].to_excel(writer, sheet_name="15_phase_inconsistente", index=False)

        # Qualidade
        qualidade["nulos"].to_excel(writer, sheet_name="16_nulos", index=False)
        qualidade["constantes"].to_excel(writer, sheet_name="17_variaveis_constantes", index=False)
        qualidade["qualidade_numerica"].to_excel(writer, sheet_name="18_qualidade_numerica", index=False)
        qualidade["outliers_detalhe"].to_excel(writer, sheet_name="19_outliers_exemplos", index=False)
        qualidade["repacking_resumo"].to_excel(writer, sheet_name="20_repacking_resumo", index=False)
        qualidade["valores_repacking"].to_excel(writer, sheet_name="21_repacking_valores", index=False)

        # Leakage
        leakage["leakage_colunas"].to_excel(writer, sheet_name="22_leakage_colunas", index=False)
        leakage["diagnostico_status"].to_excel(writer, sheet_name="23_leakage_status", index=False)

        # Recomendações
        recomendacoes.to_excel(writer, sheet_name="24_recomendacoes", index=False)


def cria_vars_momen(data):
    """
    Função para criar variáveis com janelas(momentum)
    """
    print("Criando variáveis com momentum...")
    df = data.copy()
    
    for janela in range(6, 13, 3):
        print(f"JANELA: {janela}")
        df[f'mom_receita_{janela}'] = (
            df
            .groupby('sk_produto_case')
            ['ind_vlr_receita_real_corrigido']
            .pct_change(janela)
        )
    
        df[f'mom_receita_dia_{janela}'] = (
            df
            .groupby('sk_produto_case')
            ['ind_vlr_receita_real_dia_corrigido']
            .pct_change(janela)
        )
    
        ## CPF
        df[f'mom_cpfs_total_{janela}'] = (
            df
            .groupby('sk_produto_case')
            ['ind_cpfs_total']
            .pct_change(janela)
        )
        df[f'mom_cpfs_novos_{janela}'] = (
            df
            .groupby('sk_produto_case')
            ['ind_cpfs_novos']
            .pct_change(janela)
        )
    
    
        ##CONVERSAO
        df[f'mom_conversao_{janela}'] = (
            df
            .groupby('sk_produto_case')
            ['tx_conversao_pdv']
            .pct_change(janela)
        )
    
    
        ## RUPTURA
        df[f'mom_ruptura_{janela}'] = (
            df
            .groupby('sk_produto_case')
            ['ind_vlr_ruptura']
            .pct_change(janela)
        )
    
        ## MKT
        df[f'mom_mkt_{janela}'] = (
            df
            .groupby('sk_produto_case')
            ['vlr_investimento_mkt_direto']
            .pct_change(janela)
        )
    df = df.replace([np.inf, -np.inf],np.nan)
    return df


##MÉDIA MOVEIS
def criar_med_moveis(data):
    """
    Função para criar variáveis com média móvel
    """
    print("Criando variáveis com média móvel...")
    df = data.copy()
    
    for janela in range(6, 13, 3):
        df[f'mm_receita_{janela}'] = (
            df
            .groupby('sk_produto_case')
            ['ind_vlr_receita_real_corrigido']
            .transform(
                lambda s: s.rolling(
                    janela,
                    min_periods=1
                ).mean()
            )
        )
    df = df.replace([np.inf, -np.inf],np.nan)
    return df



def calcular_slope(x):
    y = x.values

    if len(y) < 2:
        return np.nan

    x_axis = np.arange(len(y))

    slope = np.polyfit(
        x_axis,
        y,
        1
    )[0]

    return slope

def executa_slope(df):
    print("Criando variáveis com SLOPE...")
    data = df.copy()
    for x in range(6,13,3):
        data[f"slope_receita_{x}"] = (
            data.groupby("sk_produto_case")["ind_vlr_receita_real_corrigido"]
            .transform(
                lambda s:
                s.rolling(
                    window=x,
                    min_periods=2
                ).apply(
                    calcular_slope,
                    raw=False
                )
            )
        )
    return data

def calcular_slope_normalizado(x):
    y = x.values.astype(float)
    
    if len(y) < 2:
        return np.nan

    media = np.mean(y)

    if media == 0:
        return np.nan

    y = y / media

    x_axis = np.arange(len(y))

    slope = np.polyfit(
        x_axis,
        y,
        1
    )[0]

    return slope

def executa_slope_normalizado(df):
    print("Criando variáveis com SLOPE NORMALIZADO...")
    data = df.copy()
    for x in range(6,13,3):
        data[f"slope_receita_norm_{x}"] = (
            data.groupby("sk_produto_case")["ind_vlr_receita_real_corrigido"]
            .transform(
                lambda s:
                s.rolling(
                    window=x,
                    min_periods=2
                ).apply(
                    calcular_slope_normalizado,
                    raw=False
                )
            )
        )
    return data

def calcula_std_receita(df):
    print("Criando variáveis com STD RECEITA...")
    data = df.copy()
    for x in range(6,13,3):
        data[f"std_receita_{x}"] = ( data
                                        .groupby("sk_produto_case")["ind_vlr_receita_real_corrigido"]
                                        .transform(
                                            lambda s:
                                            s.rolling(
                                                window=x,
                                                min_periods=2
                                            ).std()
                                        )
                                    )
    return data

def calcula_cv_receita(df):
    print("Criando variáveis com COEF VAR...")
    data = df.copy()
    
    for x in range(6,13,3):
        media_receita = (
            data.groupby("sk_produto_case")["ind_vlr_receita_real_corrigido"]
            .transform(
                lambda s:
                s.rolling(
                    window=x,
                    min_periods=2
                ).mean()
            )
        )
        
        data[f"cv_receita_{x}"] = (data[f"std_receita_{x}"]/media_receita)
    return data
