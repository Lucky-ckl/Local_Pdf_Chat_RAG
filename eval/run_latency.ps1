# P5.5 延迟分解驱动脚本（WBS v1.0 §8 P5.5）
# 采集：④全量改造（含分阶段）与①改造前基线（仅检索总耗时）的 P50/P95。
param(
    [string]$Tag = (Get-Date -Format 'yyyyMMdd-HHmmss'),
    [string]$BaselineDir = "$env:TEMP\rag_p5_baseline",
    [int]$Repeats = 5
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$pych = Join-Path $repo '.venv\Scripts\python.exe'
$evalSet = Join-Path $repo 'eval\eval_set.json'
$corpus = Join-Path $repo 'eval\corpus'

function Invoke-Latency {
    param([string]$Group, [string]$Script, [string]$IndexDir)
    $outDir = Join-Path $repo "eval\results\${Tag}_$Group"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $args = @($Script, '--group', $Group, '--eval-set', $evalSet, '--corpus', $corpus,
              '--index-dir', $IndexDir, '--repeats', "$Repeats",
              '--out', (Join-Path $outDir 'latency.json'))
    Write-Host "== latency $Group =="
    & $pych @args *>&1 | Tee-Object -FilePath (Join-Path $outDir 'latency.log')
}

# ④ 全量改造
$env:RERANK_METHOD = 'cross_encoder'
$env:RERANK_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
$env:OCR_ENABLED = 'true'
Invoke-Latency -Group 'full' -Script 'eval\collect_latency.py' -IndexDir 'eval\index\full'

# ① 改造前基线（旧代码）
$baseEval = Join-Path $BaselineDir 'eval'
New-Item -ItemType Directory -Force -Path $baseEval | Out-Null
Copy-Item (Join-Path $repo 'eval\collect_latency.py') (Join-Path $baseEval 'collect_latency.py') -Force
Copy-Item (Join-Path $repo 'eval\run_eval.py') (Join-Path $baseEval 'run_eval.py') -Force
Invoke-Latency -Group 'baseline' -Script (Join-Path $baseEval 'collect_latency.py') `
    -IndexDir (Join-Path $BaselineDir 'eval\index\baseline')

Write-Host "done: eval\results\${Tag}_* latency"
