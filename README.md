# Ising-model-codex

開放境界の横磁場 Ising 鎖を quimb の二サイト DMRG で計算します。

\[
H=-J\sum_{i=0}^{N-2}Z_iZ_{i+1}-h\sum_{i=0}^{N-1}X_i
\]

ここで X, Z は固有値 ±1 の Pauli 行列です。既定値は臨界点 J=h=1、N=32。

## 実行

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python ising_dmrg.py --validate
```

このクラウド環境では `/workspace/ising-venv/bin/python` を利用できます。
パラメータの変更例：`python ising_dmrg.py --sites 64 --J 1 --h 0.5 --output results_h05`。

## 出力

`results/ising_dmrg.png` と PDF に、切断位置 l のエントロピーと鎖中央から右側への相関関数を描きます。
エントロピーは左側 l サイトの縮約密度行列から S(l)=-Tr(ρ log ρ) として計算し、単位は nats（自然対数）です。
相関関数は ⟨Z_i Z_j⟩、連結相関関数は ⟨Z_i Z_j⟩−⟨Z_i⟩⟨Z_j⟩。
サイト番号は 0 始まりで、既定の参照サイトは i=15 です。開放境界では相関は距離だけでなく位置にも依存します。

- `entropy.csv`: 全切断位置のエントロピー
- `correlation.csv`: 参照サイトからの距離と相関関数
- `zz_matrix.csv`, `connected_zz_matrix.csv`: 全サイト対の相関行列（対角成分も含む）
- `magnetization_z.csv`: 各サイトの ⟨Z_i⟩
- `correlation_matrix.png`: 相関行列の図
- `summary.json`: エネルギー、スイープ履歴、検証結果

最大許容ボンド次元は 16→32→64→128、切り捨て閾値は 10⁻¹²、エネルギー収束閾値は 10⁻¹⁰、最大20スイープです。
`--validate` は8サイトの独立した厳密対角化に対して、エネルギー、全切断のエントロピー、全 ZZ 相関を検証します。
有限サイズの計算であり、熱力学極限の結果ではありません。
